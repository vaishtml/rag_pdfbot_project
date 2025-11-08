# fastapi_app.py
import os, tempfile
from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document

app = FastAPI(title="RAG PDF Bot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],allow_credentials=True,
    allow_methods=["*"],allow_headers=["*"],
)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore(file_path):
    ext = file_path.split(".")[-1].lower()
    loader = PyPDFLoader(file_path) if ext == "pdf" else TextLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = [Document(page_content=t) for d in docs for t in splitter.split_text(d.page_content)]

    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(splits, embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 4})

def get_llm():
    # use the key that was passed from Streamlit
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key for Gemini.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = PromptTemplate(
    template=("Use ONLY the context below to answer.\n"
              "If not found, say 'I don't know.'\n\n"
              "Context:\n{context}\n\nQuestion:\n{question}"),
    input_variables=["context", "question"],
)

@app.post("/ask")
async def ask_question(
    file: UploadFile,
    question: str = Form(...),
    x_api_key: str = Header(None)
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key header")
    os.environ["GOOGLE_API_KEY"] = x_api_key  

    with tempfile.NamedTemporaryFile(delete=False, suffix="."+file.filename.split(".")[-1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    retriever = build_vectorstore(tmp_path)
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

    res = qa.invoke({"query": question})
    return {"answer": res["result"]}

