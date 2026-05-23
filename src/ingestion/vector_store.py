import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def add_documents(self, documents: List[Document]) -> Chroma:
        """Adds documents to the Chroma vector store and persists it."""
        if not documents:
            return None
            
        print(f"Adding {len(documents)} documents to ChromaDB at {self.persist_directory}...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vector_store.persist()
        print("Documents added and ChromaDB persisted.")
        return vector_store

    def load_vector_store(self) -> Chroma:
        """Loads an existing Chroma vector store."""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"ChromaDB not found at {self.persist_directory}. Please create it first.")
        
        print(f"Loading ChromaDB from {self.persist_directory}...")
        vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )
        print("ChromaDB loaded.")
        return vector_store

    def get_all_documents(self) -> List[Document]:
        """Retrieves all documents from the Chroma vector store."""
        try:
            vector_store = self.load_vector_store()
            # chroma_db.get() returns a dict with 'documents', 'metadatas', 'ids', etc.
            data = vector_store.get()
            
            documents = []
            if data and 'documents' in data:
                for i in range(len(data['documents'])):
                    documents.append(Document(
                        page_content=data['documents'][i],
                        metadata=data['metadatas'][i] if data['metadatas'] else {}
                    ))
            return documents
        except FileNotFoundError:
            return []
