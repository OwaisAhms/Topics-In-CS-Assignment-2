import faiss
import json
import numpy as np
from typing import List, Tuple


class RAGStore:
def __init__(self, dim: int):
self.dim = dim
self.index = faiss.IndexFlatIP(dim) # cosine via normalized vectors
self.metadatas: List[dict] = []


def add(self, vectors: List[List[float]], metadatas: List[dict]):
arr = np.array(vectors, dtype='float32')
# normalize for cosine similarity
faiss.normalize_L2(arr)
self.index.add(arr)
self.metadatas.extend(metadatas)


def save(self, index_path: str, meta_path: str):
faiss.write_index(self.index, index_path)
with open(meta_path, 'w', encoding='utf-8') as f:
json.dump(self.metadatas, f, ensure_ascii=False, indent=2)


@classmethod
def load(cls, index_path: str, meta_path: str):
idx = faiss.read_index(index_path)
dim = idx.d
store = cls(dim)
store.index = idx
with open(meta_path, 'r', encoding='utf-8') as f:
store.metadatas = json.load(f)
return store


def query(self, vector: List[float], k: int = 4) -> List[Tuple[float, dict]]:
import numpy as np
arr = np.array([vector], dtype='float32')
faiss.normalize_L2(arr)
D, I = self.index.search(arr, k)
results = []
for score, idx in zip(D[0], I[0]):
if idx < 0 or idx >= len(self.metadatas):
continue
results.append((float(score), self.metadatas[idx]))
return results