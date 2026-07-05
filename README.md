# AI Knowledge Assistant

Ask a company's own documents a question and get an instant, cited answer — instead of searching a shared drive.

**Problem:** Employees waste time re-finding answers that already live in a handbook, policy doc, or product PDF.
**Solution:** Upload documents once; a local LLM answers questions using only that content, with the source shown.
**Result:** Lookup time drops from minutes to seconds, and nothing leaves the company's machine (everything runs locally via Ollama).

## Architecture

```
 PDFs / .docx / .txt
        │
        ▼
  ingest.py  ──►  chunk + embed  ──►  ChromaDB (local vector store)
        │
        ▼
  app.py (Streamlit)
        │
        ├─► retrieve top-k relevant chunks for the question
        └─► send question + chunks to a local LLM (via Ollama)
                       │
                       ▼
              answer + cited source shown to the user
```

## Features
- Drag-and-drop document upload (PDF, DOCX, TXT)
- Local embeddings + local LLM — no data leaves the machine
- Answers show which document/section they came from
- Swap models by changing one config value (llama3, qwen2.5, gemma2, phi3, deepseek — anything pulled in Ollama)

## Installation

1. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull llama3.1
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put your documents in `sample_docs/` (a sample HR policy is included).
4. Build the index:
   ```bash
   python ingest.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Configuration

Edit the top of `app.py`:
```python
OLLAMA_MODEL = "llama3.1"       # any model you've pulled with `ollama pull`
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 4                 # how many chunks to retrieve per question
```

## Future improvements
- Add per-user access control so different roles see different documents
- Add a feedback button ("was this answer correct?") to catch bad retrievals
- Swap ChromaDB for a hosted vector DB for multi-user deployments
- Add conversation memory for follow-up questions

## Demo script (for a client call)
1. Upload the client's actual FAQ/handbook live on the call.
2. Ask the exact question their team gets asked most.
3. Show the answer arriving in seconds, with the source line highlighted.
