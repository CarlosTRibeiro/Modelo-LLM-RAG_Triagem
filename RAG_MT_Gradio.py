# app_gradio.py

import gradio as gr
from pathlib import Path
import warnings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore", category=DeprecationWarning)

# —————————————————————————————————
# 1) Prepara o pipeline RAG uma vez, fora da função
# —————————————————————————————————

# Carrega todos os .txt do teu knowledge base
txt_loader = DirectoryLoader(
    "./TXT", glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding":"utf-8"}
)
documents = txt_loader.load()

# Parte em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Cria embeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Carrega o prompt externo
legal = Path("LegalNotice.txt").read_text(encoding="utf-8")
priority  = Path("priority_rules.txt").read_text(encoding="utf-8")
template_text = legal + "\n\n" + priority + "\n\n" + Path("prompt_template.txt").read_text(encoding="utf-8")
prompt = PromptTemplate(input_variables=["context","question"], template=template_text)

# Instancia o LLM e o RetrievalQA chain
llm = ChatOllama(model="llama2", streaming=False)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False,
)

# —————————————————————————————————
# 2) Função que o Gradio vai chamar por cada input
# —————————————————————————————————
def triage_assistant(question: str) -> str:
    if not question.strip():
        return "Please describe the patient's complaint or symptoms."
    return qa_chain.run(question)

# —————————————————————————————————
# 3) Monta a interface Gradio
# —————————————————————————————————
with gr.Blocks(title="🩺 Manchester Triage Assistant") as demo:
    gr.Markdown("# 🩺 Manchester Triage Assistant")
    with gr.Row():
        txt = gr.Textbox(
            lines=2
        )
        btn = gr.Button("Submit", variant="primary")

    output = gr.Markdown()

    btn.click(fn=triage_assistant, inputs=txt, outputs=output)

# —————————————————————————————————
# 4) Arranca a app
# —————————————————————————————————
if __name__ == "__main__":
    demo.launch()
