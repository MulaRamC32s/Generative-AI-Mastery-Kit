import streamlit as st
import os
import sys
from typing import List, Optional

# Add the root directory to path to allow importing core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import RAGEngine, Document
from core.agent_manager import AgentManager
from dotenv import load_dotenv

load_dotenv()

# Streamlit UI Configuration
st.set_page_config(
    page_title="Generative AI Mastery Kit",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session State for Engines
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
    # Pre-populate with some sample data for demonstration
    mock_docs = [
        Document(page_content="The Generative AI Mastery Kit is designed for AI engineers to build production-grade apps.", metadata={"source": "Internal Docs"}),
        Document(page_content="RAG systems combine retrieval and generation to provide more accurate context-aware responses.", metadata={"source": "AI Wiki"}),
        Document(page_content="Agents are autonomous entities that use LLMs as reasoning engines to call external tools.", metadata={"source": "Agent Lab"})
    ]
    st.session_state.rag_engine.ingest_documents(mock_docs)

if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.radio("Select Mode", ["RAG Assistant", "Tool-Using Agent"])
    
    st.divider()
    
    st.subheader("RAG Engine Stats")
    if st.session_state.rag_engine.vector_store:
        st.success(f"Vector Store Active")
        st.info("Ingested 3 Sample Docs")
    else:
        st.warning("Vector Store Empty")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main UI
st.title("🤖 Generative AI Mastery Kit")
st.markdown("""
Welcome to the professional-grade Generative AI toolkit. This interface demonstrates the power of 
**Retrieval-Augmented Generation** and **Autonomous LLM Agents**.
""")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handle
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Input based on Mode
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            if mode == "RAG Assistant":
                response = st.session_state.rag_engine.query(prompt)
                full_response = response["answer"]
                
                # Show source docs if available
                if response.get("source_documents"):
                    with st.expander("Show Sources"):
                        for idx, doc in enumerate(response["source_documents"]):
                            st.write(f"Source {idx+1}: {doc.metadata.get('source', 'Unknown')}")
                            st.caption(doc.page_content)
                            
            else:  # Tool-Using Agent
                try:
                    full_response = st.session_state.agent_manager.run(prompt)
                except Exception as e:
                    full_response = f"Agent Error: {str(e)}"

            st.markdown(full_response)
            
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.divider()
st.caption("Powered by LangChain, FAISS, and OpenAI. Built for professional Generative AI development.")
