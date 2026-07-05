"""
ingest.py — reads every file in sample_docs/, splits it into overlapping
chunks, embeds each chunk, and stores it in a local ChromaDB collection.

Run this once whenever documents are added or changed:
    python ingest.py
"""
import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document as DocxDocument

DOCS_DIR = "sample_docs"
DB_DIR = "chroma_db"
COLLECTION_NAME = "company_knowledge"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks
EMBED_MODEL = "all-MiniLM-L6-v2"

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(path):
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)

READERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
}

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, breaking on paragraph boundaries where possible."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # try to end on a paragraph or sentence boundary
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start:
            end = boundary + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end) if end - overlap <= start else end - overlap
    return [c for c in chunks if c]

def load_documents(docs_dir):
    """Returns list of (source_filename, chunk_text) tuples for every supported file."""
    records = []
    paths = glob.glob(os.path.join(docs_dir, "**", "*"), recursive=True)
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        reader = READERS.get(ext)
        if not reader or not os.path.isfile(path):
            continue
        try:
            text = reader(path)
        except Exception as e:
            print(f"  ! skipped {path}: {e}")
            continue
        chunks = chunk_text(text)
        for chunk in chunks:
            records.append((os.path.basename(path), chunk))
    return records

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    print(f"Reading documents from ./{DOCS_DIR} ...")
    records = load_documents(DOCS_DIR)
    if not records:
        print(f"No documents found in {DOCS_DIR}/. Add PDFs, DOCX, or TXT files and re-run.")
        return

    print(f"Found {len(records)} chunks across the document set. Embedding...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)

    # start fresh each run so stale chunks don't linger
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)
    ids = [f"chunk-{i}" for i in range(len(records))]
    documents = [r[1] for r in records]
    metadatas = [{"source": r[0]} for r in records]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Done. Indexed {len(records)} chunks into '{COLLECTION_NAME}' at ./{DB_DIR}")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
