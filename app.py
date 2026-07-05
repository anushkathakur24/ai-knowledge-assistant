"""
app.py — a Streamlit chat UI that answers questions about a company's own
documents, using local embeddings (ChromaDB) and a local LLM (Ollama).

Run:
    streamlit run app.py

Prerequisite:
    python ingest.py   (must be run first, and re-run after adding documents)
"""
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import ollama

# ---- Configuration --------------------------------------------------------
OLLAMA_MODEL = "llama3.1"        # change to any model you've pulled: `ollama pull <model>`
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_DIR = "chroma_db"
COLLECTION_NAME = "company_knowledge"
TOP_K = 4                         # how many chunks to retrieve per question

SYSTEM_PROMPT = (
    "You are an internal knowledge assistant. Answer ONLY using the "
    "provided context from company documents. If the answer isn't in the "
    "context, say you don't have that information and suggest who to ask. "
    "Be concise and cite the source filename in parentheses at the end."
)

# ---- Setup -----------------------------------------------------------------
st.set_page_config(page_title="Knowledge Assistant", page_icon="📄", layout="centered")

@st.cache_resource(show_spinner=False)
def get_collection():
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        return None

def retrieve_context(collection, question, k=TOP_K):
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

def build_prompt(question, context_pairs):
    context_block = "\n\n".join(
        f"[Source: {meta.get('source', 'unknown')}]\n{doc}" for doc, meta in context_pairs
    )
    return (
        f"Context from company documents:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )

def ask_ollama(question, context_pairs):
    prompt = build_prompt(question, context_pairs)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]

# ---- UI ---------------------------------------------------------------------
st.title("📄 Company Knowledge Assistant")
st.caption("Ask a question. The answer comes only from documents that have been indexed — nothing leaves this machine.")

collection = get_collection()
if collection is None:
    st.warning(
        "No index found yet. Add documents to `sample_docs/` and run "
        "`python ingest.py` before asking questions."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask about company policy, a product, or a process...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            context_pairs = retrieve_context(collection, question)
            if not context_pairs:
                answer = "I couldn't find anything relevant in the indexed documents."
            else:
                answer = ask_ollama(question, context_pairs)
        st.markdown(answer)
        with st.expander("Sources used"):
            for doc, meta in context_pairs:
                st.markdown(f"**{meta.get('source', 'unknown')}**")
                st.caption(doc[:300] + ("..." if len(doc) > 300 else ""))
    st.session_state.history.append(("assistant", answer))
