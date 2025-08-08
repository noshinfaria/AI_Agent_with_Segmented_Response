from openai import OpenAI
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("static/sentence_ending.html")


def is_sentence_ending(text: str):
    # You can improve this later for better logic
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


@app.get("/chat")
async def chat(prompt: str):
    async def event_generator():
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        buffer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                buffer += delta.content
                if is_sentence_ending(buffer.strip()):
                    yield {
                        "event": "message",
                        "data": buffer.strip()
                    }
                    buffer = ""
                    await asyncio.sleep(1.5)

        if buffer.strip():
            yield {
                "event": "message",
                "data": buffer.strip()
            }

    return EventSourceResponse(event_generator())
