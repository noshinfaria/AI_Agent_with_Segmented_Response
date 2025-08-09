## Chunking for Topic Transitions
This project uses OpenAI GPT-4 or LangChain streaming pipelines to detect topic transitions in long texts and dynamically stream segmented output to the user using FastAPI and Server-Sent Events (SSE).

It focuses on semantic chunking — breaking text only where the topic changes, not at every sentence, with two main implementations:

OpenAI Direct Streaming – Manually buffers tokens, detects sentences, and calls GPT for semantic segmentation.

LangChain Streaming – Uses custom callbacks to detect paragraph or sentence boundaries in real-time without manual buffering.

### File Structure
```graphql
AI_Agent_with_Segmented_Response/
├── static/
│   ├── openai_chunk.html       # Frontend UI for interacting with the AI agent for openai based chunking
│   └── sentence_ending.html    # Frontend UI for interacting with the AI agent for sentence ending based    chunking
├── Openai/
│   ├── openai_chunk.py         # FastAPI backend with GPT-4 chunking and streaming logic
│   └── sentence_ending.py      # FastAPI backend with sentence ending based 
├── Langchain/
│   ├── paragraph_based_streaming.py         # FastAPI backend with GPT-4 chunking and streaming logic
│   └── sentence_based_streaming.py      # FastAPI backend with sentence ending based 
├── .gitignore                  # Specifies files/folders to ignore in Git
├── README.md                   # Project documentation and setup instructions
├── env.example                 # Example environment config (to be copied as `.env`)
├── openai_chunk.py             # FastAPI backend with GPT-4 chunking and streaming logic
├── requirements.txt            # Project dependencies
├── sentence_ending.py          # FastAPI backend with sentence ending based
```

### Chunking Modes
1. OpenAI Semantic Chunking (Topic Transitions)
- Uses OpenAI GPT-4 to insert [TOPIC SHIFT] markers only when topics actually change.
- Buffers tokens (~50 tokens at a time) before sending to GPT for segmentation.
- Respects both token limits and sentence boundaries.
- Best for structured, topic-based documents.

2. LangChain Paragraph-Based Streaming
- Uses LangChain callbacks to detect paragraph boundaries in real-time.
- No manual token buffering — paragraphs are streamed as they complete.
- Great for long-form narratives, articles, or essays where paragraph separation matters more than sentence-level cuts.

3. LangChain Sentence-Based Streaming
- Uses LangChain callbacks to detect sentence endings in real-time.
- Sentences are emitted instantly as they finish without waiting for large buffers.
- Ideal for chat-like experiences where responsiveness is key.

## When to Use Each Mode
Mode	Technology Used	Unit of Chunking	Pros	Cons	Best For
OpenAI Semantic Chunking	OpenAI GPT-4 + FastAPI	Topic Segments	High-level semantic awareness, true topic shifts	More latency (needs batching + GPT calls)	Reports, research papers
LangChain Paragraph-Based	LangChain + FastAPI	Paragraphs	Preserves natural paragraph flow, minimal delay	Less semantic filtering	Articles, blogs, storytelling
LangChain Sentence-Based	LangChain + FastAPI	Sentences	Very fast, responsive	No topic-level awareness	Chatbots, live Q&A


### Keypoints
- Accepts prompt from the user.
- Streams responses in real-time.
- Supports topic-based, paragraph-based, and sentence-based chunking modes.
- Works with both OpenAI GPT-4 API and LangChain streaming callbacks.
- SSE ensures smooth delivery to the frontend.

### Key Technologies
FastAPI – API server and SSE response handling.
OpenAI GPT-4 – For semantic chunking (topic segmentation).
LangChain – Sentence & paragraph detection via callbacks
nltk – For sentence tokenization (punkt).
tiktoken – For accurate token counting (matching OpenAI models).
dotenv – For environment configuration.
sse_starlette – For real-time streaming to the frontend.

## How It Works
### OpenAI Mode
- Accumulate tokens (~50 tokens)
- Pass buffer to GPT-4 for [TOPIC SHIFT] detection
- Stream chunks to frontend

### LangChain Sentence Mode
- Stream tokens directly from model
- Callback detects sentence-ending punctuation
- Emit completed sentence instantly

### LangChain Paragraph Mode
- Stream tokens directly from model
- Callback detects paragraph breaks (\n\n)
- Emit completed paragraph instantly

### OpenAI Sementic Chunking (Step-by-Step)
1. Setup and Tokenizer

```python
tokenizer = tiktoken.encoding_for_model("gpt-4")
```
We use tiktoken to measure how many tokens are accumulated. This is crucial because OpenAI models have token limits, and semantic chunking requires batching by token count — not just by sentences.

2. Semantic Chunking Function

```python
async def find_semantic_chunks(text_block: str) -> list[str]
```
This function calls GPT-4 with a prompt that instructs it to insert [TOPIC SHIFT] markers only when there's a real shift in topic, not just at sentence boundaries.

The response is parsed as a JSON list of strings, like:

```json
[
  "Intro to the story.",
  "[TOPIC SHIFT] A new development happens.",
  "Continued discussion.",
  "[TOPIC SHIFT] Conclusion."
]
```

3. Streaming Chat with Token Batching
The /chat endpoint streams the chat response token-by-token using GPT-4’s streaming API.
Instead of evaluating each sentence, it buffers sentences into blocks of ~50 tokens:

```python
token_limit = 50
```
Once the buffer is full, it is passed to find_semantic_chunks().
This approach ensures:
Performance: fewer API calls (compared to per-sentence segmentation).
Context: GPT can recognize higher-level structure when given more context.

4. Sentence-Level Token Management

```python
sent_tokenize(buffer)
```
Since we don’t want to cut in the middle of a sentence, we:
- Split the buffer into sentences.
- Add sentences one by one into a block.
- Flush the block when it exceeds the token limit.

This respects both token limits and sentence boundaries, preserving semantic coherence.

5. Handling Topic Shifts
If a segment begins with [TOPIC SHIFT], it indicates a transition.

In that case:
We flush the accumulated content before the shift. Output the [TOPIC SHIFT] line separately. Start a new accumulation after the shift.

```python
if chunk.startswith("[TOPIC SHIFT]"):
    if text_accumulator.strip():
        yield f"{text_accumulator.strip()}"
        text_accumulator = ""
    yield f"{chunk}"
```

6. Final Flush
Once the stream ends, any remaining buffer is flushed one last time and processed for semantic chunks.

```python
if buffer.strip():
    chunks = await find_semantic_chunks(buffer)
```
This ensures no text is left behind unprocessed.
