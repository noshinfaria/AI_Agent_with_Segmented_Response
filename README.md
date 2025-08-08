## Semantic Chunking for Topic Transitions
This project uses OpenAI's GPT-4 to detect topic transitions in long texts and dynamically streams segmented output to the user using FastAPI and Server-Sent Events (SSE). It focuses on semantic chunking — breaking text only where the topic changes, not at every sentence.

### File Structure
```graphql
AI_Agent_with_Segmented_Response/
├── static/
│   ├── openai_chunk.html       # Frontend UI for interacting with the AI agent for openai based chunking
│   └── sentence_ending.html    # Frontend UI for interacting with the AI agent for sentence ending based    chunking
├── .gitignore                  # Specifies files/folders to ignore in Git
├── README.md                   # Project documentation and setup instructions
├── env.example                 # Example environment config (to be copied as `.env`)
├── openai_chunk.py             # FastAPI backend with GPT-4 chunking and streaming logic
├── requirements.txt            # Project dependencies
├── sentence_ending.py          # FastAPI backend with sentence ending based
```

### Keypoints
- Accepts prompt from the user.
- Streams the response from GPT in real-time.
- Groups the streamed tokens into semantically meaningful chunks.
- Uses GPT-4 again to identify [TOPIC SHIFT] markers for actual transitions in the content.
- Outputs the result incrementally via a web interface.

### Key Technologies
FastAPI – API server and SSE response handling.
OpenAI GPT-4 – For semantic chunking (topic segmentation).
nltk – For sentence tokenization (punkt).
tiktoken – For accurate token counting (matching OpenAI models).
dotenv – For environment configuration.
sse_starlette – For real-time streaming to the frontend.

### How It Works (Step-by-Step)
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
