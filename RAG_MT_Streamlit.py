# app.py
import streamlit as st
from pathlib import Path
import warnings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------------------------
# Helpers: cache para não recarregar a cada interação
# ----------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    # Carrega todos os .txt
    txt_loader = DirectoryLoader(
        "./TXT",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = txt_loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

@st.cache_resource(show_spinner=False)
def build_chain(_vectorstore):
    # Lê o prompt de disco
    legal = Path("LegalNotice.txt").read_text(encoding="utf-8")
    priority  = Path("priority_rules.txt").read_text(encoding="utf-8")
    template_text = legal + "\n\n" + priority + "\n\n" + Path("prompt_template.txt").read_text(encoding="utf-8")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template_text
    )

    # Monta o chain
    llm = ChatOllama(model="llama2", streaming=False)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.set_page_config(page_title="Manchester Triage Assistant", layout="wide")
st.title("🩺 Manchester Triage Assistant")

# Constrói vectorstore e chain (apenas na primeira vez)
vectorstore = build_vectorstore()
qa_chain = build_chain(vectorstore)

# Caixa de texto para criar nova pergunta
user_question = st.text_input("Describe the patient's complaint or symptoms:", key="input")

if st.button("Submit") and user_question:
    with st.spinner("🧠 Thinking..."):
        answer = qa_chain.run(user_question)
    st.markdown("**Assistant:**")
    st.write(answer)
