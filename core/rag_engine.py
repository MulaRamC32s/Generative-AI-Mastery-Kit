import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field

class RAGConfig(BaseModel):
    """Configuration for the RAG Engine."""
    model_name: str = Field(default="gpt-3.5-turbo")
    embedding_model: str = Field(default="text-embedding-3-small")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k: int = Field(default=3)

class RAGEngine:
    """A professional-grade Retrieval-Augmented Generation (RAG) Engine."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self.llm = ChatOpenAI(model_name=self.config.model_name, temperature=0)
        self.vector_store = None
        self.qa_chain = None

    def ingest_documents(self, documents: List[Document]):
        """Ingest documents into the FAISS vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vector_store.add_documents(splits)
            
        self._initialize_qa_chain()
        return f"Successfully ingested {len(splits)} document chunks."

    def _initialize_qa_chain(self):
        """Initialize the retrieval chain."""
        if self.vector_store:
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

    def query(self, query: str) -> dict:
        """Query the RAG engine for an answer."""
        if not self.qa_chain:
            return {"result": "RAG Engine not initialized. Please ingest documents first."}
            
        response = self.qa_chain.invoke({"query": query})
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }

    def save_index(self, path: str):
        """Persist the FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)

    def load_index(self, path: str):
        """Load the FAISS index from disk."""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            self._initialize_qa_chain()
            return True
        return False

# Example usage for mocking/testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Mocking data for a quick test
    mock_docs = [
        Document(page_content="The Generative AI Mastery Kit is an advanced toolkit for LLM development.", metadata={"source": "doc1"}),
        Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors.", metadata={"source": "doc2"})
    ]
    
    engine = RAGEngine()
    engine.ingest_documents(mock_docs)
    res = engine.query("What is FAISS?")
    print(res["answer"])
