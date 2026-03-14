# Generative AI Mastery Kit

A professional-grade repository demonstrating robust Generative AI architectures, including Retrieval-Augmented Generation (RAG) and Tool-Using Agents.

## 🏗 Architecture Overview

This project implements a modular, enterprise-ready architecture for modern LLM applications:

1.  **RAG Pipeline**: Leverages LangChain for document ingestion, chunking, and semantic search via FAISS. It provides contextually relevant answers by grounding the LLM in external knowledge.
2.  **Agentic Framework**: Uses an ReAct-style agent capable of calling external tools (e.g., calculations, web search, database queries) to solve complex, multi-step tasks.
3.  **Streamlit UI**: A polished, interactive interface for seamless human-AI interaction.

## 🚀 Features

-   **Semantic Search**: High-performance vector retrieval with FAISS.
-   **Tool-Using Agents**: Robust LLM agents with pre-defined tools.
-   **Modern Python Tooling**: Built with LangChain, OpenAI API, and Pydantic for strong typing and validation.
-   **Scalable & Modular**: Clean separation of concerns between core logic and the application layer.

## 🛠 Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Generative-AI-Mastery-Kit.git
    cd Generative-AI-Mastery-Kit
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_openai_key_here
    ```

5.  **Run the application**:
    ```bash
    streamlit run app/streamlit_ui.py
    ```

## 📂 Project Structure

```text
Generative-AI-Mastery-Kit/
├── app/
│   └── streamlit_ui.py     # Frontend Streamlit application
├── core/
│   ├── rag_engine.py      # RAG Pipeline and Vector DB logic
│   └── agent_manager.py    # LLM Agent and Tool definitions
├── .gitignore             # Standard git exclusions
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

## ✨ Future Enhancements

-   [ ] Support for Open-Source LLMs (via Ollama or vLLM).
-   [ ] Integration with Managed Vector Databases (Pinecone, Weaviate).
-   [ ] Enhanced Agent Tooling (SQL Database access, Code Interpreter).
-   [ ] Multi-modal support for Image/Audio processing.

---
Developed with ❤️ by Senior Generative AI Engineers.
