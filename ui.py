"""Chainlit chat UI that talks to the FastAPI backend. Run: chainlit run ui.py"""
import os

import chainlit as cl
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "Hi! Ask a question about your documents, or ask me to calculate or plot "
            "numbers from them (for example: *plot revenue by quarter as a bar chart*)."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("id")
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{API_URL}/chat", json={"question": message.content, "session_id": session_id}
            )
            resp.raise_for_status()
            data = resp.json()

            elements = []
            for name in data["artifacts"]:
                file_resp = await client.get(f"{API_URL}/artifacts/{name}")
                if file_resp.status_code != 200:
                    continue
                if name.lower().endswith(".png"):
                    elements.append(cl.Image(name=name, content=file_resp.content, display="inline"))
                else:
                    elements.append(cl.File(name=name, content=file_resp.content))
    except Exception as exc:  # network down, API error, ...
        await cl.Message(content=f"Sorry, the backend did not answer ({type(exc).__name__}).").send()
        return

    text = data["answer"]
    if data["sources"]:
        lines = [f"[{s['n']}] {s['source']}, page {s['page']}" for s in data["sources"]]
        text += "\n\n**Sources**\n" + "\n".join(f"- {line}" for line in lines)
    text += f"\n\n*route: {data['intent']} - {data['seconds']}s*"
    await cl.Message(content=text, elements=elements).send()
