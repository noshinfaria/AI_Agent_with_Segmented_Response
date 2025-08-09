import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.callbacks.base import AsyncCallbackHandler

load_dotenv()

# --------------------------
# Custom Callback for Paragraphs
# --------------------------
class ParagraphCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.buffer = ""

    async def on_llm_new_token(self, token: str, **kwargs):
        self.buffer += token
        # Detect paragraph break
        if "\n\n" in self.buffer:
            paragraph, self.buffer = self.buffer.split("\n\n", 1)
            if paragraph.strip():
                await self.queue.put(paragraph.strip())

    async def on_llm_end(self, *args, **kwargs):
        if self.buffer.strip():
            await self.queue.put(self.buffer.strip())
        await self.queue.put(None)  # Signal completion


# --------------------------
# FastAPI Setup
# --------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("static/sentence_ending.html")


@app.get("/chat")
async def chat(prompt: str):
    async def event_generator():
        queue = asyncio.Queue()
        callback = ParagraphCallbackHandler(queue)

        llm = ChatOpenAI(
            model="gpt-4o",
            streaming=True,
            callbacks=[callback],
            temperature=0
        )

        # Start LLM in background
        asyncio.create_task(llm.ainvoke([{"role": "user", "content": prompt}]))

        while True:
            paragraph = await queue.get()
            if paragraph is None:
                break
            yield {
                "event": "message",
                "data": paragraph
            }
            # await asyncio.sleep(1)  # Delay between paragraphs if desired

    return EventSourceResponse(event_generator())
