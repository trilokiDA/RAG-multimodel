import os
from typing import List
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.ingestion.vector_store import VectorStoreManager

class HybridRetriever:
    def __init__(self, chroma_persist_dir: str = "./chroma_db", embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.chroma_persist_dir = chroma_persist_dir
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        self.vector_store_manager = VectorStoreManager(
            embedding_model_name=self.embedding_model_name,
            persist_directory=self.chroma_persist_dir
        )
        self.chroma_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

    def initialize_retrievers(self, documents: List[Document]):
        """
        Initializes the dense (Chroma) and sparse (BM25) retrievers.
        If ChromaDB exists, it loads it. Otherwise, it creates it if documents are provided.
        """
        # Initialize dense retriever (ChromaDB)
        try:
            vector_store = self.vector_store_manager.load_vector_store()
            self.chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            print("ChromaDB retriever loaded from disk.")
        except FileNotFoundError:
            if not documents:
                print("Warning: ChromaDB not found and no documents provided for initialization.")
                return # Can't initialize retrievers
            print("ChromaDB not found, creating a new one from provided documents...")
            self.chroma_retriever = self.vector_store_manager.add_documents(documents).as_retriever(search_kwargs={"k": 5})
        
        # Initialize sparse retriever (BM25)
        # BM25 must be initialized with the full set of documents
        if not documents:
            print("Attempting to load documents from ChromaDB for BM25 initialization...")
            documents = self.vector_store_manager.get_all_documents()
            
        if not documents:
            print("Warning: No documents available to initialize BM25 retriever. Hybrid retrieval will be limited.")
            # We can't initialize the ensemble retriever without both
            return
            
        print(f"Initializing BM25 with {len(documents)} documents.")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 5 # Retrieve 5 documents

        # Initialize Ensemble Retriever
        if self.chroma_retriever and self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.chroma_retriever],
                weights=[0.5, 0.5]
            )
            print("Hybrid (Ensemble) retriever successfully initialized.")
        else:
            print("Failed to initialize one or more retrievers. Hybrid retrieval disabled.")

    def get_ensemble_retriever(self) -> EnsembleRetriever:
        """Returns the initialized EnsembleRetriever."""
        if self.ensemble_retriever is None:
            raise RuntimeError("Retrievers not initialized. Call initialize_retrievers first.")
        return self.ensemble_retriever

if __name__ == "__main__":
    # This is a minimal example. In a real scenario, you'd load actual documents
    # and run the ingestion pipeline first.
    print("--- Testing HybridRetriever ---")
    
    # Create a dummy ChromaDB for testing if it doesn't exist
    from src.ingestion.loader import DocumentLoader
    dummy_loader = DocumentLoader()
    dummy_docs_text, dummy_multimodal_docs = dummy_loader.load_and_split("data/example.txt") # Use the dummy file created by pipeline.py
    
    # Ensure data directory and dummy files exist
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/example.txt"):
        with open("data/example.txt", "w") as f:
            f.write("This is an example text document about artificial intelligence and machine learning. It contains information about neural networks.")
    if not os.path.exists("data/example_bm25.txt"):
        with open("data/example_bm25.txt", "w") as f:
            f.write("This is another document for BM25 testing, focusing on AI algorithms and deep learning.")

    # Load all documents for BM25
    all_dummy_docs = []
    text_chunks, multimodal_docs = dummy_loader.load_and_split("data/example.txt")
    all_dummy_docs.extend(text_chunks)
    all_dummy_docs.extend(multimodal_docs)
    text_chunks_bm25, multimodal_docs_bm25 = dummy_loader.load_and_split("data/example_bm25.txt")
    all_dummy_docs.extend(text_chunks_bm25)
    all_dummy_docs.extend(multimodal_docs_bm25)


    # Initialize and test the retriever
    hybrid_retriever_instance = HybridRetriever()
    hybrid_retriever_instance.initialize_retrievers(all_dummy_docs)

    query = "What is artificial intelligence?"
    print(f"Querying: {query}")
    results = hybrid_retriever_instance.get_ensemble_retriever().invoke(query)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:100]}...")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
