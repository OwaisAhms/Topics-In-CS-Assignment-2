# Talk-to-Your-Docs (Ollama RAG)

A Retrieval-Augmented Generation (RAG) API powered by [Ollama](https://ollama.com) and FAISS for querying PDFs using LLMs. Upload PDFs, build an index, and ask questions with context-aware answers.  

## **Getting Started**

### to run the project**

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit values if needed

python -m app.pdf_loader --pdf_dir ./pdfs --out index.faiss --meta meta.json

docker-compose up --build

uvicorn app.main:app --reload --port 8000

# query example

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main topic of chapter 1?"}'

# to run offline tests
cd tests
python eval.py
