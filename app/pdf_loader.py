import argparse
from pathlib import Path
import json
from PyPDF2 import PdfReader
from tqdm import tqdm
from app.embeddings import embed_texts
from app.rag_store import RAGStore


CHUNK_SIZE = 800
CHUNK_OVERLAP = 200




def split_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
words = text.split()
out = []
i = 0
while i < len(words):
chunk = words[i:i+size]
out.append(" ".join(chunk))
i += size - overlap
return out




def load_pdfs(pdf_dir: str) -> list[dict]:
docs = []
pdf_dir = Path(pdf_dir)
for p in pdf_dir.glob('**/*.pdf'):
reader = PdfReader(str(p))
text = []
for page in reader.pages:
try:
text.append(page.extract_text() or "")
except Exception:
continue
full = "\n".join(text)
docs.append({"path": str(p), "text": full, "name": p.name})
return docs




def build_index(pdf_dir: str, out_index: str, out_meta: str):
docs = load_pdfs(pdf_dir)
chunks = []
metas = []
for d in docs:
for i, chunk in enumerate(split_text(d['text'])):
metas.append({
'source': d['name'],
'chunk_id': i,
'text': chunk[:2000]
})
chunks.append(chunk)
print(f"Embedding {len(chunks)} chunks...")
vectors = embed_texts(chunks)
dim = len(vectors[0])
store = RAGStore(dim)
store.add(vectors, metas)
store.save(out_index, out_meta)
print('Saved index and meta')




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--pdf_dir', required=True)
parser.add_argument('--out', dest='out_index', required=True)
parser.add_argument('--meta', dest='out_meta', required=True)
args = parser.parse_args()
build_index(args.pdf_dir, args.out_index, args.out_meta)