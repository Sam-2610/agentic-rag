"""Measure the agent on 17 test messages (needs your LLM key and the sample report indexed).

    python eval_agent.py

Reports router accuracy, answer correctness, code-agent success, latency and LLM calls.
"""
import csv
import re
import time
import uuid

import agent
import llm

PAUSE = 4  # seconds between questions, to be gentle with free-tier rate limits

# (question, expected_intent, keywords, min_keywords, kind)
# kind: "text" = answer must contain min_keywords of the keywords | "refuse" = must say not found
#       "chart" = a .png must be produced | "any" = only the intent is checked
CASES = [
    ("hello", "chat", [], 0, "any"),
    ("thanks a lot", "chat", [], 0, "any"),
    ("What can you do?", "chat", ["document", "calculate", "plot", "question", "chart"], 1, "text"),
    ("Who is the CEO of Northwind Traders?", "retrieve", ["alvarez"], 1, "text"),
    ("What risks does the report mention?", "retrieve", ["supply", "currency", "supplier"], 2, "text"),
    ("How many employees did the company have at the end of the year?", "retrieve", ["240"], 1, "text"),
    ("Where did the company open a new distribution center in Q3?", "retrieve", ["rotterdam"], 1, "text"),
    ("What revenue growth does management expect for 2026?", "retrieve", ["12"], 1, "text"),
    ("Who won the 2023 cricket world cup?", "retrieve", [], 0, "refuse"),
    ("What is the total revenue across all regions in Q3?", "analyze", ["1365"], 1, "text"),
    ("Which region had the highest percentage growth from Q1 to Q4?", "analyze", ["east"], 1, "text"),
    ("What was the total annual revenue for all regions?", "analyze", ["5305"], 1, "text"),
    ("Calculate the average quarterly revenue for the North region", "analyze", ["472.5"], 1, "text"),
    ("Plot revenue by quarter as a bar chart", "analyze", [], 0, "chart"),
    ("Make a line chart of revenue by region over the four quarters", "analyze", [], 0, "chart"),
]
# Two turns in the SAME session: tests conversation memory
MEMORY_CASES = [
    ("My favorite region is East, remember that.", "chat", [], 0, "any"),
    ("Which region did I say is my favorite?", "chat", ["east"], 1, "text"),
]


def passed(out, kind, keywords, need):
    text = out["answer"].lower().replace(",", "")
    if kind == "refuse":
        return out["answer"] == agent.NOT_FOUND_TEXT or "couldn't find" in text
    if kind == "chart":
        return any(a.endswith(".png") for a in out["artifacts"])
    if kind == "text":
        return sum(k in text for k in keywords) >= need
    return True


def main():
    counter = {"n": 0}
    real_chat = llm.chat

    def counting_chat(*args, **kwargs):
        counter["n"] += 1
        return real_chat(*args, **kwargs)

    llm.chat = counting_chat

    memory_id = f"eval-mem-{uuid.uuid4().hex[:6]}"
    plan = [(c, f"eval-{uuid.uuid4().hex[:6]}") for c in CASES] + [(c, memory_id) for c in MEMORY_CASES]

    rows, skipped_in_row = [], 0
    for (q, want_intent, kws, need, kind), session in plan:
        before = counter["n"]
        out = agent.ask(q, session)
        if out["answer"] == agent.BUSY_TEXT:
            skipped_in_row += 1
            print(f"SKIP  (LLM busy)  | {q[:55]}")
            if skipped_in_row >= 3:
                print("\nToo many rate-limit errors. Wait, or switch provider in .env, then run again.")
                break
            continue
        skipped_in_row = 0
        ok = passed(out, kind, kws, need)
        rows.append({
            "question": q, "expected": want_intent, "intent": out["intent"], "pass": ok,
            "seconds": out["seconds"], "llm_calls": counter["n"] - before, "trace": " | ".join(out["trace"]),
            "answer": out["answer"],
        })
        print(f"{'PASS' if ok else 'FAIL'}  {out['intent']:8} {out['seconds']:5.1f}s  calls={counter['n'] - before}  | {q[:50]}")
        time.sleep(PAUSE)

    if not rows:
        return
    n = len(rows)
    intent_ok = sum(r["intent"] == r["expected"] for r in rows)
    answer_ok = sum(r["pass"] for r in rows)
    no_retrieval = sum("retrieve:" not in r["trace"] for r in rows)
    analyze = [r for r in rows if r["expected"] == "analyze"]
    first_try = sum("attempt 1): ok" in r["trace"] for r in analyze)
    eventually = sum(bool(re.search(r"attempt \d\): ok", r["trace"])) for r in analyze)
    cited = [r for r in rows if r["expected"] == "retrieve" and r["pass"] and "[" in r["answer"]]

    print("\n=========== RESULTS ===========")
    print(f"Questions run:            {n} of {len(plan)}")
    print(f"Router accuracy:          {intent_ok}/{n} = {intent_ok / n:.0%}")
    print(f"Answer correctness:       {answer_ok}/{n} = {answer_ok / n:.0%}")
    if analyze:
        print(f"Code agent, 1st attempt:  {first_try}/{len(analyze)}")
        print(f"Code agent, after fixes:  {eventually}/{len(analyze)}")
    print(f"Retrieval skipped by router: {no_retrieval}/{n} messages (a no-router baseline retrieves for all {n})")
    print(f"LLM calls per message:    {sum(r['llm_calls'] for r in rows) / n:.1f} average")
    for intent in ("chat", "retrieve", "analyze"):
        part = [r["seconds"] for r in rows if r["intent"] == intent]
        if part:
            print(f"Avg latency, {intent:8}:   {sum(part) / len(part):.1f}s")
    print(f"Retrieve answers with citations: {len(cited)}")

    with open("eval_agent_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Saved eval_agent_results.csv")


if __name__ == "__main__":
    main()
