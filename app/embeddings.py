import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

def embed_texts(texts):
    out = []
    for t in texts:
        r = requests.post(OLLAMA_EMBED_URL, json={
            "model": EMBED_MODEL,
            "input": t
        })
        out.append(r.json()["embedding"])
    return out

def embed_text(text):
    return embed_texts([text])[0]