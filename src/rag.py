import os
import streamlit as st
from typing import TypedDict, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from src.prompts import RAG_PROMPT

# absolute path — always saves next to app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")


class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    sources: List[str]


@st.cache_resource
def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


@st.cache_resource
def get_vectorstore(kb_path: str, api_key: str):
    embeddings = get_embeddings(api_key)

    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    loader = DirectoryLoader(kb_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\nTOPIC:", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def make_retrieve_node(vectorstore):
    def retrieve(state: RAGState) -> RAGState:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["question"])
        sources = list(set([
            os.path.basename(doc.metadata.get("source", "unknown"))
            for doc in docs
        ]))
        return {**state, "context": docs, "sources": sources}
    return retrieve


def make_generate_node(api_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)
    prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["context", "question"])

    def generate(state: RAGState) -> RAGState:
        context_text = "\n\n".join([doc.page_content for doc in state["context"]])
        filled_prompt = prompt.format(context=context_text, question=state["question"])
        response = llm.invoke(filled_prompt)
        answer = response.content if isinstance(response.content, str) else response.content[0]["text"]
        return {**state, "answer": answer}
    return generate


@st.cache_resource
def build_rag_chain(_vectorstore, api_key: str):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(_vectorstore))
    graph.add_node("generate", make_generate_node(api_key))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def get_answer(chain, question: str) -> dict:
    result = chain.invoke({"question": question, "context": [], "answer": "", "sources": []})
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def initialize_rag(api_key: str, kb_path: str = "knowledge_base"):
    vectorstore = get_vectorstore(kb_path, api_key)
    return build_rag_chain(vectorstore, api_key)