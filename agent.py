"""The agent: a LangGraph state machine.

            +--> chat ------------------------------------+
  router ---+                                             |
            +--> retrieve --> answer ---------------------+--> finalize --> END
            |        |                                    |
            +--> (analyze) --> write_code --> run_code --(ok)--> summarize
                                    ^            |  (error, attempts left)
                                    +-- fix_code-+  (error, no attempts left) --> answer (fallback)
"""
import json
import operator
import re
import time
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

import llm
import sandbox
import store
from config import MAX_CODE_ATTEMPTS

NOT_FOUND_TEXT = "I couldn't find this in the documents."
BUSY_TEXT = "The AI service is busy right now. Please try again in a minute."

# ---------------------------------------------------------------- prompts
ROUTER_SYSTEM = (
    "Classify the user's new message into exactly one intent:\n"
    '- "retrieve": a question to be answered from the uploaded documents (facts, explanations, summaries).\n'
    '- "analyze": needs a calculation, a comparison of numbers, a table or a chart built from numbers in the documents.\n'
    '- "chat": greetings, thanks, questions about this conversation or about what you can do, '
    "or anything that needs no documents.\n"
    'Reply with ONLY JSON: {"intent": "retrieve" | "analyze" | "chat", "reason": "<a few words>"}'
)

ANSWER_SYSTEM = (
    "You answer questions using ONLY the numbered context passages. Do not add facts that are not in them. "
    "Answer in 2-5 short sentences (or a short list if asked for a list) and cite passages like [1] or [2]. "
    "You may use the conversation history to understand references such as 'it' or 'that region'. "
    "If the passages do not contain the answer, reply with exactly: NOT_FOUND"
)

CHAT_SYSTEM = (
    "You are a friendly assistant for a document question-answering system. Keep replies short. "
    "You can answer questions from the user's documents with citations, and calculate or plot numbers found in them. "
    "Use the conversation history when the user refers to earlier messages."
)

CODE_SYSTEM = (
    "You write short Python programs that analyze data found in document text.\n"
    "Rules:\n"
    "- The variable CONTEXT (a string with the retrieved document text) is already defined. Do not read files.\n"
    "- Parse the numbers you need from CONTEXT with plain Python (str.split, re, float). "
    "Do NOT use pandas.read_csv or any read_* function.\n"
    "- Allowed imports only: pandas, numpy, matplotlib, math, statistics, json, re, datetime, collections, "
    "itertools, io, csv, decimal.\n"
    "- Print every result you need with print().\n"
    "- For a chart use matplotlib.pyplot, add a title and axis labels, and save it with "
    "plt.savefig('chart.png', bbox_inches='tight'). Never call plt.show().\n"
    "- Do not use open(), eval(), exec(), os, sys or subprocess.\n"
    "Return ONLY the code in one ```python block."
)

FIX_SYSTEM = (
    "You fix a Python program that failed. Keep the same rules as before: CONTEXT is already defined, "
    "parse it with plain Python (no read_* functions, no open/os/sys), allowed imports are pandas, numpy, "
    "matplotlib, math, statistics, json, re, datetime, collections, itertools, io, csv, decimal, "
    "print every result, and save charts with plt.savefig('chart.png', bbox_inches='tight'). "
    "Return ONLY the corrected full program in one ```python block."
)

SUMMARY_SYSTEM = (
    "You explain the result of a data analysis to the user in 1-4 short sentences. "
    "Use ONLY the numbers in the program output. If a chart file was created, mention it. "
    "You may cite source passages like [1]. Do not invent numbers."
)

GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thank you so much|thanks a lot|thx|ok|okay|bye|"
    r"good morning|good afternoon|good evening)[\s!.]*$",
    re.I,
)
ANALYZE_WORDS = re.compile(
    r"\b(plot|chart|graph|visuali[sz]e|histogram|calculate|compute|sum|total|average|mean|median|"
    r"percentage|percent|growth|compare|trend|rank|sort)\b",
    re.I,
)


# ---------------------------------------------------------------- state
class State(TypedDict, total=False):
    question: str
    history: Annotated[list, operator.add]  # grows across turns (conversation memory)
    intent: str
    docs: list
    code: str
    run: dict
    attempts: int
    answer: str
    sources: list
    artifacts: list
    trace: list


class Intent(BaseModel):
    intent: Literal["retrieve", "analyze", "chat"]
    reason: str = ""


# ---------------------------------------------------------------- helpers
def recent(state, n=6):
    return [{"role": m["role"], "content": m["content"]} for m in state.get("history", [])[-n:]]


def docs_text(docs):
    return "\n\n".join(f"[{i + 1}] ({d['source']}, page {d['page']})\n{d['text']}" for i, d in enumerate(docs))


def to_sources(docs):
    return [{"n": i + 1, "source": d["source"], "page": d["page"], "score": d["score"]} for i, d in enumerate(docs)]


def add_trace(state, line):
    return state.get("trace", []) + [line]


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def retrieval_query(state):
    """Short follow-ups ("and in Q4?") get the previous user question prepended."""
    q = state["question"]
    earlier = [m for m in state.get("history", []) if m["role"] == "user"]
    if earlier and len(q.split()) < 6:
        return f"{earlier[-1]['content']} {q}"
    return q


def classify(question, history):
    """Ask the LLM for the intent; fall back to simple rules if it fails or returns garbage."""
    convo = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-4:]) or "(none)"
    reply = llm.chat(
        [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"Recent conversation:\n{convo}\n\nNew message: {question}"},
        ],
        task="router",
    )
    if reply:
        match = re.search(r"\{.*\}", reply, re.S)
        if match:
            try:
                parsed = Intent(**json.loads(match.group(0)))
                return parsed.intent, parsed.reason or "llm"
            except (ValidationError, ValueError, TypeError):
                pass
    return ("analyze" if ANALYZE_WORDS.search(question) else "retrieve"), "fallback rules"


# ---------------------------------------------------------------- nodes
def router_node(state: State) -> dict:
    q = state["question"]
    if GREETING.match(q):
        intent, why = "chat", "greeting shortcut, no LLM call"
    else:
        intent, why = classify(q, state.get("history", []))
    return {
        "intent": intent, "docs": [], "code": "", "run": {}, "attempts": 0,
        "answer": "", "sources": [], "artifacts": [],
        "trace": [f"router: {intent} ({why})"],
    }


def chat_node(state: State) -> dict:
    messages = [{"role": "system", "content": CHAT_SYSTEM}, *recent(state), {"role": "user", "content": state["question"]}]
    reply = llm.chat(messages, task="chat")
    return {"answer": (reply or BUSY_TEXT).strip(), "trace": add_trace(state, "chat: replied without documents")}


def retrieve_node(state: State) -> dict:
    docs = store.search(retrieval_query(state))
    return {"docs": docs, "trace": add_trace(state, f"retrieve: {len(docs)} chunks")}


def after_retrieve(state: State) -> str:
    return "write_code" if state["intent"] == "analyze" and state["docs"] else "answer"


def answer_node(state: State) -> dict:
    docs = state.get("docs", [])
    trace = state.get("trace", [])
    if state.get("run") and not state["run"].get("ok"):
        trace = trace + ["answer: analysis failed, falling back to a text answer"]
    if not docs:
        return {"answer": NOT_FOUND_TEXT, "sources": [], "trace": trace + ["answer: no relevant chunks"]}

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        *recent(state),
        {"role": "user", "content": f"Context:\n{docs_text(docs)}\n\nQuestion: {state['question']}"},
    ]
    reply = llm.chat(messages, task="answer")
    if reply is None:
        return {"answer": BUSY_TEXT, "sources": [], "trace": trace + ["answer: LLM unavailable"]}
    if "NOT_FOUND" in reply:
        return {"answer": NOT_FOUND_TEXT, "sources": [], "trace": trace + ["answer: model said not found"]}

    sources = to_sources(docs)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", reply)}
    sources = [s for s in sources if s["n"] in cited] or sources
    return {"answer": reply.strip(), "sources": sources, "trace": trace + ["answer: grounded reply"]}


def write_code_node(state: State) -> dict:
    messages = [
        {"role": "system", "content": CODE_SYSTEM},
        {"role": "user", "content": f"Question: {state['question']}\n\nCONTEXT contains:\n{docs_text(state['docs'])}"},
    ]
    reply = llm.chat(messages, task="write_code")
    return {"code": extract_code(reply or ""), "trace": add_trace(state, "write_code: program written")}


def run_code_node(state: State) -> dict:
    result = sandbox.run(state["code"], docs_text(state["docs"]))
    attempts = state["attempts"] + 1
    note = "ok" if result["ok"] else "error: " + (result["error"].strip().splitlines() or ["unknown"])[-1][:120]
    return {"run": result, "attempts": attempts, "trace": add_trace(state, f"run_code (attempt {attempts}): {note}")}


def after_run(state: State) -> str:
    if state["run"]["ok"]:
        return "summarize"
    return "fix_code" if state["attempts"] < MAX_CODE_ATTEMPTS else "answer"


def fix_code_node(state: State) -> dict:
    messages = [
        {"role": "system", "content": FIX_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {state['question']}\n\nProgram:\n```python\n{state['code']}\n```\n\n"
                f"Error:\n{state['run']['error']}\n\nCONTEXT contains:\n{docs_text(state['docs'])}"
            ),
        },
    ]
    reply = llm.chat(messages, task="fix_code")
    code = extract_code(reply) if reply else state["code"]
    return {"code": code, "trace": add_trace(state, "fix_code: program corrected")}


def summarize_node(state: State) -> dict:
    run = state["run"]
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {state['question']}\n\nProgram output:\n{run['stdout']}\n\n"
                f"Files created: {', '.join(run['artifacts']) or 'none'}\n\nSource passages:\n{docs_text(state['docs'])}"
            ),
        },
    ]
    reply = llm.chat(messages, task="summarize")
    answer = reply.strip() if reply else f"Result:\n{run['stdout']}"
    return {
        "answer": answer, "sources": to_sources(state["docs"]), "artifacts": run["artifacts"],
        "trace": add_trace(state, "summarize: result explained"),
    }


def finalize_node(state: State) -> dict:
    return {"history": [{"role": "user", "content": state["question"]}, {"role": "assistant", "content": state["answer"]}]}


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(State)
    for name, fn in [
        ("router", router_node), ("chat", chat_node), ("retrieve", retrieve_node), ("answer", answer_node),
        ("write_code", write_code_node), ("run_code", run_code_node), ("fix_code", fix_code_node),
        ("summarize", summarize_node), ("finalize", finalize_node),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["intent"], {"chat": "chat", "retrieve": "retrieve", "analyze": "retrieve"})
    g.add_conditional_edges("retrieve", after_retrieve, {"write_code": "write_code", "answer": "answer"})
    g.add_edge("write_code", "run_code")
    g.add_conditional_edges("run_code", after_run, {"summarize": "summarize", "fix_code": "fix_code", "answer": "answer"})
    g.add_edge("fix_code", "run_code")
    for name in ("chat", "answer", "summarize"):
        g.add_edge(name, "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())  # MemorySaver = per-session conversation memory


graph = build_graph()


def ask(question: str, session_id: str = "default") -> dict:
    t0 = time.time()
    out = graph.invoke({"question": question}, config={"configurable": {"thread_id": session_id}})
    return {
        "answer": out["answer"], "intent": out["intent"], "sources": out["sources"],
        "artifacts": out["artifacts"], "trace": out["trace"], "seconds": round(time.time() - t0, 2),
    }
