# RAG_MT_with_Memory.py

from pathlib import Path
import warnings

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_static_files():
    """
    Carrega:
      - prompt_template.txt   (raiz)
      - priority_rules.txt    (raiz)
      - Glossary.txt          (raiz)
      - todos os cenários em ./scenarios/
    Retorna: (core_template, priority_rules, glossary, scenarios_concatenados).
    """
    root = Path(".")
    prompt_path = root / "prompt_template.txt"
    priority_path = root / "priority_rules.txt"
    glossary_path = root / "Glossary.txt"
    scenarios_dir = root / "scenarios"

    if not prompt_path.exists() or not priority_path.exists() or not glossary_path.exists():
        raise FileNotFoundError(
            "Certifica-te que 'prompt_template.txt', 'priority_rules.txt' e 'Glossary.txt' existem na raiz do projeto."
        )

    if not scenarios_dir.exists():
        raise FileNotFoundError("Cria a pasta 'scenarios/' na raiz e coloca aí os ficheiros de cenário (*.txt).")

    core_template = prompt_path.read_text(encoding="utf-8")
    priority_rules = priority_path.read_text(encoding="utf-8")
    glossary = glossary_path.read_text(encoding="utf-8")

    txt_files = list(scenarios_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError("A pasta 'scenarios/' precisa conter pelo menos um ficheiro .txt de cenário.")

    scenarios = ""
    for scenario_file in txt_files:
        scenarios += scenario_file.read_text(encoding="utf-8") + "\n\n"

    return core_template, priority_rules, glossary, scenarios


def build_prompt_template(core_template: str,
                          priority_rules: str,
                          glossary: str,
                          scenarios: str) -> PromptTemplate:
    """
    Substitui apenas {glossary}, {priority_rules} e {scenarios} em core_template,
    mantendo {context} e {question} intactos para preenchimento posterior.
    """
    filled = core_template.replace("{glossary}", glossary) \
                           .replace("{priority_rules}", priority_rules) \
                           .replace("{scenarios}", scenarios)
    return PromptTemplate(
        input_variables=["context", "question"],
        template=filled
    )


def build_faiss_index(glossary_path: str = "./Glossary.txt",
                      scenarios_folder: str = "./scenarios",
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200,
                      embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                      index_path: str = "faiss_index_Triage_Manchester") -> FAISS:
    """
    Carrega:
      - Glossary.txt (raiz) → TextLoader
      - Todos os ficheiros .txt em scenarios_folder → DirectoryLoader
    Divide tudo em chunks, gera embeddings e cria (ou recarrega) um índice FAISS.
    Salva localmente em index_path.
    """
    glossary_file = Path(glossary_path)
    if not glossary_file.exists():
        raise FileNotFoundError(f"O ficheiro '{glossary_path}' não foi encontrado.")
    glossary_docs = TextLoader(glossary_path, encoding="utf-8").load()

    scenarios_dir = Path(scenarios_folder)
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"A pasta '{scenarios_folder}' não foi encontrada.")
    scenario_files = list(scenarios_dir.glob("*.txt"))
    if not scenario_files:
        raise FileNotFoundError(f"A pasta '{scenarios_folder}' não contém ficheiros .txt.")
    scenario_docs = DirectoryLoader(
        scenarios_folder,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    ).load()

    all_docs = glossary_docs + scenario_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        raise RuntimeError("Não foram gerados chunks ao dividir os documentos para FAISS.")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore


def initialize_chain(filled_prompt: PromptTemplate,
                     index: FAISS,
                     llm_model: str = "llama2",
                     temperature: float = 0.0,
                     top_p: float = 1.0,
                     top_k: int = 1,
                     k_retriever: int = 5) -> ConversationalRetrievalChain:
    """
    Cria e retorna um ConversationalRetrievalChain com memória simples,
    configurado com o LLM (ChatOllama) e o índice FAISS.
    """
    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    retriever = index.as_retriever(search_kwargs={"k": k_retriever})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": filled_prompt}
    )


def main():
    # 1) Carregar ficheiros estáticos e de conhecimento
    core_template, priority_rules, glossary, scenarios = load_static_files()

    # 2) Montar o PromptTemplate
    prompt = build_prompt_template(core_template, priority_rules, glossary, scenarios)

    # 3) Construir ou carregar o índice FAISS (Glossary.txt + cenários em ./scenarios/)
    vectorstore = build_faiss_index(
        glossary_path="./Glossary.txt",
        scenarios_folder="./scenarios"
    )

    # 4) Inicializar o chain com memória (k_retriever=5)
    qa_chain = initialize_chain(prompt, vectorstore, k_retriever=5)

    # 5) Loop de interação
    print("💬 RAG Manchester Triage System with Memory (type 'exit' to quit)")
    chat_history = []  # Mantém uma lista de mensagens para cada turno

    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        
            # Passar 'question' e 'chat_history' na chamada
        result = qa_chain({"question": user_input, "chat_history": chat_history})
        answer = result["answer"]
        chat_history = result["chat_history"]  # agora é lista de mensagens
        

        print("🧠 Assistant:", answer)


if __name__ == "__main__":
    main()
