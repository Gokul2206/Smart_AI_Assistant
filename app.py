import streamlit as st
import os
from dotenv import load_dotenv
from src.rag import initialize_rag, get_answer
from src.tasks import get_model_recommendation, explain_ml_code

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="ML Buddy", page_icon="🤖", layout="centered")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.rag_chain is None:
    with st.spinner("Loading knowledge base..."):
        try:
            kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
            st.session_state.rag_chain = initialize_rag(api_key, kb_path)
        except Exception as e:
            st.error(str(e))

with st.sidebar:
    st.title("🤖 ML Buddy")
    st.caption("AI/ML Knowledge Assistant")
    st.divider()
    mode = st.radio("", ["💬 Chat", "🎯 Model Recommender", "🔍 Code Explainer"], label_visibility="collapsed")
    st.divider()
    if st.session_state.rag_chain:
        st.success("Ready")
    if st.session_state.chat_history:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

if mode == "💬 Chat":
    st.header("Ask anything about AI/ML")

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if item.get("sources"):
                st.caption("Sources: " + " · ".join(item["sources"]))

    question = st.chat_input("Ask a question...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = get_answer(st.session_state.rag_chain, question)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        st.caption("Sources: " + " · ".join(result["sources"]))
                    st.session_state.chat_history.append(result | {"question": question})
                except Exception as e:
                    st.error(str(e))

elif mode == "🎯 Model Recommender":
    st.header("Model Recommender")
    st.caption("Describe your problem and get the best model suggestion.")
    problem = st.text_area("", placeholder="e.g. Binary classification, 5000 rows, 10 features, need interpretability", height=130)
    if st.button("Recommend →", type="primary"):
        if not problem.strip():
            st.warning("Describe your problem first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    st.markdown(get_model_recommendation(problem, api_key))
                except Exception as e:
                    st.error(str(e))

elif mode == "🔍 Code Explainer":
    st.header("Code Explainer")
    st.caption("Paste any Python ML code and get a clear explanation.")
    code = st.text_area("", placeholder="# Paste your ML code here...", height=250)
    if st.button("Explain →", type="primary"):
        if not code.strip():
            st.warning("Paste some code first.")
        else:
            with st.spinner("Reading..."):
                try:
                    st.markdown(explain_ml_code(code, api_key))
                except Exception as e:
                    st.error(str(e))