from openai import OpenAI
import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import nltk
import json
import tiktoken
from nltk.tokenize import sent_tokenize

# Setup
load_dotenv()
nltk.download("punkt_tab")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.encoding_for_model("gpt-4")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("static/openai_chunk.html")


def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


async def find_semantic_chunks(text_block: str) -> list[str]:
    """
    Use GPT to mark topic shifts in a block of text.
    Output is a list of strings, some starting with [TOPIC SHIFT].
    """
    prompt = f"""
You are an expert in text structure. Given a story, segment it at major topic transitions.
Only add '[TOPIC SHIFT]' at real topic changes — not every sentence.

Output format: A JSON list of strings like:
[
  "This is the intro.",
  "[TOPIC SHIFT] Something new happens.",
  "Continuation of same topic.",
  "[TOPIC SHIFT] Final twist."
]

Now segment this story:
{text_block}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You segment the story only at real topic transitions, using [TOPIC SHIFT] as a marker.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return parsed

    except Exception as e:
        print("Error during semantic chunking:", e)
        return [text_block]  # fallback


@app.get("/chat")
async def chat(prompt: str):
    async def event_generator():
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        buffer = ""
        token_limit = 50
        text_accumulator = ""

        # async def flush_if_big_enough():
        #     nonlocal buffer, text_accumulator
        #     if num_tokens(buffer) >= token_limit:
        #         chunks = await find_semantic_chunks(buffer)
        #         buffer = ""
        #         for chunk in chunks:
        #             if chunk.startswith("[TOPIC SHIFT]"):
        #                 if text_accumulator.strip():
        #                     yield f"{text_accumulator.strip()}\n\n"
        #                     await asyncio.sleep(1.0)
        #                     text_accumulator = ""
        #                 yield f"{chunk}\n\n"
        #                 await asyncio.sleep(0.5)
        #             else:
        #                 text_accumulator += " " + chunk.strip()

        async def flush_if_big_enough():
            nonlocal buffer, text_accumulator
            if num_tokens(buffer) >= token_limit:
                sentences = sent_tokenize(buffer)
                buffer = ""

                sentence_block = ""
                for sentence in sentences:
                    tentative_block = (sentence_block + " " + sentence).strip()
                    if num_tokens(tentative_block) > token_limit:
                        # This block is too big, so flush the existing one
                        if sentence_block:
                            chunks = await find_semantic_chunks(sentence_block)
                            for chunk in chunks:
                                if chunk.startswith("[TOPIC SHIFT]"):
                                    if text_accumulator.strip():
                                        yield f"{text_accumulator.strip()}\n\n"
                                        await asyncio.sleep(1.0)
                                        text_accumulator = ""
                                    yield f"{chunk}\n\n"
                                    await asyncio.sleep(0.5)
                                else:
                                    text_accumulator += " " + chunk.strip()
                            sentence_block = sentence
                        else:
                            # Single sentence too long, force flush it
                            chunks = await find_semantic_chunks(sentence)
                            for chunk in chunks:
                                if chunk.startswith("[TOPIC SHIFT]"):
                                    if text_accumulator.strip():
                                        yield f"{text_accumulator.strip()}\n\n"
                                        await asyncio.sleep(1.0)
                                        text_accumulator = ""
                                    yield f"{chunk}\n\n"
                                    await asyncio.sleep(0.5)
                                else:
                                    text_accumulator += " " + chunk.strip()
                            sentence_block = ""
                    else:
                        sentence_block = tentative_block

                buffer = sentence_block  # carry over any unflushed remainder

        for chunk in stream: 
            delta = chunk.choices[0].delta
            if delta and delta.content:
                buffer += delta.content
                async for output in flush_if_big_enough():
                    yield output

        # Final flush
        if buffer.strip():
            chunks = await find_semantic_chunks(buffer)
            for chunk in chunks:
                if chunk.startswith("[TOPIC SHIFT]"):
                    if text_accumulator.strip():
                        yield f"{text_accumulator.strip()}\n\n"
                        await asyncio.sleep(1.0)
                        text_accumulator = ""
                    yield f"{chunk}\n\n"
                    await asyncio.sleep(0.5)
                else:
                    text_accumulator += " " + chunk.strip()

        if text_accumulator.strip():
            yield f"{text_accumulator.strip()}\n\n"

    return EventSourceResponse(event_generator())
