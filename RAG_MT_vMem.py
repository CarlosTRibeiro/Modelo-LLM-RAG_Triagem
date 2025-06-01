from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Step 1: Load the document
txt_loader = DirectoryLoader(
    "./TXT",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
txt_docs = txt_loader.load()

documents = txt_docs

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Step 3: Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create vector store with FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Optional: Save index locally
vectorstore.save_local("faiss_index_Triage_Manchester")

# Step 5: Use Ollama model (e.g., mistral or phi)
llm = ChatOllama(model="llama2")  # You can replace with "phi", "gemma", etc.

legal = Path("LegalNotice.txt").read_text(encoding="utf-8")
priority  = Path("priority_rules.txt").read_text(encoding="utf-8")
 # Step 5.1: Carrega o template de disco
template_text = legal + "\n\n" + priority + "\n\n" + Path("prompt_template.txt").read_text(encoding="utf-8")
# Cria o PromptTemplate usando o conteúdo do ficheiro
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template_text
)

# ———————————————

# Step 6: Create QA chain
# limita ao top-3 fragmentos
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
)



# Step 7: Query the system
print("💬 RAG Manchester Triage System (type 'exit' to quit)")


chat_history = []
while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ("exit","quit"):
        break
    result = qa_chain({
        "question": user_input,
        "chat_history": chat_history
    })
    answer = result["answer"]
    print("🧠 Assistente:", answer)
    chat_history.append((user_input, answer))
