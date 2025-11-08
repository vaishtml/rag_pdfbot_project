import streamlit as st, requests, os

st.set_page_config(page_title="RAG PDF Bot", layout="centered")
st.title("📘 RAG PDF Bot — Streamlit + FastAPI")

API_URL = "http://localhost:8000/ask"
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
question = st.text_input("Ask your question")

if st.button("Ask") and file and question:
    with st.spinner("Processing..."):
        files = {"file": (file.name, file.getbuffer(), file.type)}
        data = {"question": question}
        headers = {"x-api-key": GOOGLE_API_KEY}
        res = requests.post(API_URL, files=files, data=data, headers=headers)
    if res.status_code == 200:
        st.success("✅ Answer")
        st.write(res.json()["answer"])
    else:
        st.error("❌ Something went wrong.")
