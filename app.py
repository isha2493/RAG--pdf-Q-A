"""Chat with your PDF.   Run with:  streamlit run app.py"""
import hashlib

import streamlit as st

from answer import MODEL, answer_question, ollama_ready
from ingest import ingest_pdf_bytes, list_documents

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_upload" not in st.session_state:
    st.session_state.last_upload = None


def reset_chat():
    st.session_state.messages = []


# ---------------- sidebar: upload + choose PDF ----------------
with st.sidebar:
    st.header("Your PDF")

    ok, problem = ollama_ready()
    if ok:
        st.caption(f"🟢 AI ready ({MODEL})")
    else:
        st.error(problem)

    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None:
        data = uploaded.getvalue()
        key = hashlib.md5(data).hexdigest()
        if st.session_state.last_upload != key:          # only process a new file once
            with st.spinner("Reading the PDF and getting it ready (first time can take a minute)..."):
                try:
                    name = ingest_pdf_bytes(data, uploaded.name)
                except Exception as e:
                    st.error(f"Could not process this PDF: {e}")
                else:
                    st.session_state.last_upload = key
                    st.session_state["doc"] = name        # select it in the box below
                    reset_chat()
                    st.success(f"Ready: {name}")

    docs = list_documents()
    if docs:
        st.selectbox("Ask questions about:", docs, key="doc", on_change=reset_chat)

doc = st.session_state.get("doc")
if not doc:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

st.caption(f"Answering only from: **{doc}**")

# ---------------- chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Where this came from"):
                for page, score, text in m["sources"]:
                    st.markdown(f"**Page {page}**")
                    st.text(text[:400])

# ---------------- new question ----------------
if question := st.chat_input("Ask a question about this PDF"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and thinking (can take 10-60 seconds on a laptop)..."):
            try:
                result = answer_question(question, doc)
            except Exception as e:
                result = {"found": False, "answer": f"Something went wrong: {e}", "sources": []}
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Where this came from"):
                for page, score, text in result["sources"]:
                    st.markdown(f"**Page {page}**")
                    st.text(text[:400])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )