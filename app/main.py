import os
import time
import json
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

from app.embeddings import embed_text
from app.rag_store import RAGStore
from app.pdf_loader import load_pdfs, split_text
from app.telemetry import log_request

load_dotenv()  # loads .env in project root when running locally

# Config (from env or defaults)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_PATH = os.getenv("OLLAMA_CHAT_PATH", "/api/chat")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}{OLLAMA_CHAT_PATH}"
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
INDEX_PATH = os.getenv("INDEX_PATH", "index.faiss")
META_PATH = os.getenv("META_PATH", "meta.json")
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "2000"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You answer user questions using the retrieved document excerpts. "
    "Cite sources like [file.pdf]. Do not hallucinate. If unclear, ask a short clarifying question."
)

INJECTION_PATTERNS = [
    "ignore previous", "disregard previous", "forget instructions", "override the", "jailbreak", "system:"
]

app = FastAPI(title="talk-to-your-docs (Ollama RAG)")

# Load index if exists
rag_store: Optional[RAGStore] = None
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    try:
        rag_store = RAGStore.load(INDEX_PATH, META_PATH)
        print(f"[startup] Loaded index: {INDEX_PATH} (dim={rag_store.dim})")
    except Exception:
        print("[startup] Failed to load index:", traceback.format_exc())
else:
    print("[startup] Index not found. Build one with the pdf loader script before querying.")


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": rag_store is not None}


def check_prompt_injection(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in INJECTION_PATTERNS)


def call_ollama_chat(messages: list[dict]) -> dict:
    """
    Call Ollama chat endpoint with the messages list (system/user/assistant roles).
    Returns the parsed JSON from Ollama (attempts to be compatible with the shape used earlier).
    """
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


@app.post("/query")
def query(q: Query):
    question = q.question.strip()
    start_ts = time.time()

    # input length guard
    if len(question) == 0:
        raise HTTPException(status_code=400, detail="Question is empty")
    if len(question) > MAX_INPUT_LEN:
        raise HTTPException(status_code=400, detail=f"Input too long (> {MAX_INPUT_LEN} chars)")

    # basic prompt-injection guard
    if check_prompt_injection(question):
        raise HTTPException(status_code=400, detail="Prompt-injection detected — refusing to run the query")

    # RAG retrieval (if index exists)
    pathway = "none"
    context_text = ""
    try:
        if rag_store:
            pathway = "RAG"
            vec = embed_text(question)  # app.embeddings -> Ollama embeddings
            results = rag_store.query(vec, k=4)
            # Join results as context, include source tags
            ctxs = []
            for score, meta in results:
                # ensure text field exists in meta
                txt = meta.get("text", "")
                src = meta.get("source", "unknown")
                ctxs.append(f"[source={src}] {txt}")
            context_text = "\n\n".join(ctxs)
    except Exception as e:
        # Retrieval failures shouldn't completely block local dev — log and continue as 'none'
        pathway = "none"
        context_text = ""
        print("[warn] RAG retrieval failed:", e)

    # Compose messages for Ollama chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
    ]

    # Call Ollama chat
    try:
        resp_json = call_ollama_chat(messages)
        # Ollama responses may vary by setup. Try common shapes:
        answer = None
        if isinstance(resp_json, dict):
            # older shape: {"message": {"content": "..."}} OR [{"message": {"content": "..."}}]
            if "message" in resp_json and isinstance(resp_json["message"], dict):
                answer = resp_json["message"].get("content")
            elif "output" in resp_json:
                # sometimes under 'output' or 'response'
                if isinstance(resp_json["output"], list) and len(resp_json["output"]) > 0 and "content" in resp_json["output"][0]:
                    answer = resp_json["output"][0]["content"]
            elif "choices" in resp_json and len(resp_json["choices"]) > 0:
                answer = resp_json["choices"][0].get("message", {}).get("content")
        if answer is None:
            # try fallback: if resp_json is a list
            if isinstance(resp_json, list) and len(resp_json) > 0 and isinstance(resp_json[0], dict):
                m = resp_json[0].get("message") or resp_json[0].get("output")
                if isinstance(m, dict):
                    answer = m.get("content") or m.get("text")
        if answer is None:
            # Fallback to stringified JSON
            answer = json.dumps(resp_json)[:2000]
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama API error: {e.response.text if e.response is not None else str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start_ts
    # Telemetry: log timestamp, pathway, latency
    try:
        log_request(question, pathway, latency, extra={"index_loaded": rag_store is not None})
    except Exception:
        print("[warn] telemetry logging failed")

    return {"answer": answer, "pathway": pathway, "latency_s": latency}