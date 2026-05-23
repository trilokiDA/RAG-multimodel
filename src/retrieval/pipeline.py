import os
from typing import List
from langchain_core.documents import Document
from src.retrieval.retrievers import HybridRetriever
from src.retrieval.reranker import DocumentReranker


class RetrievalPipeline:
    def __init__(
        self, 
        documents: List[Document], # All documents to be used for BM25 and Chroma initialization
        chroma_persist_dir: str = "./chroma_db", 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        retrieval_k: int = 10, # Number of documents to retrieve initially
        rerank_top_n: int = 3 # Number of documents to return after reranking
    ):
        self.documents = documents
        self.retrieval_k = retrieval_k
        self.rerank_top_n = rerank_top_n

        self.hybrid_retriever = HybridRetriever(chroma_persist_dir, embedding_model_name)
        self.hybrid_retriever.initialize_retrievers(documents) # Initialize with all documents
        self.ensemble_retriever = self.hybrid_retriever.get_ensemble_retriever()
        
        self.document_reranker = DocumentReranker(reranker_model_name)






    def retrieve_and_rerank(self, query: str) -> List[Document]:
        """
        Executes the full retrieval and reranking pipeline.
        """
        # Step 1: Initial retrieval using hybrid retriever
        print(f"Performing hybrid retrieval for query: '{query}'")
        retrieved_docs = self.ensemble_retriever.invoke(query)
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # Step 2: Rerank the retrieved documents
        print(f"Reranking documents...")
        reranked_docs = self.document_reranker.rerank(query, retrieved_docs, self.rerank_top_n)
        print(f"Returning top {len(reranked_docs)} reranked documents.")
        
        return reranked_docs

if __name__ == "__main__":
    print("--- Testing RetrievalPipeline ---")

    # Ensure data directory and dummy files exist for ingestion
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/example.txt"):
        with open("data/example.txt", "w") as f:
            f.write("This is an example text document about artificial intelligence and machine learning. It contains information about neural networks.")
    if not os.path.exists("data/example_bm25.txt"):
        with open("data/example_bm25.txt", "w") as f:
            f.write("This is another document for BM25 testing, focusing on AI algorithms and deep learning.")
    if not os.path.exists("data/graph_doc.pdf"):
        # This is a placeholder, real PDF with image content needed for full test
        with open("data/graph_doc.pdf", "w") as f:
            f.write("This PDF document talks about market trends and includes a graph showing growth over time.")

    # Run ingestion pipeline to create ChromaDB and get all documents
    from src.ingestion.pipeline import IngestionPipeline
    ingestion_pipeline = IngestionPipeline()
    ingestion_pipeline.run()

    # We need to load all documents again to initialize BM25 correctly
    # In a real app, this would be passed from the ingestion step
    from src.ingestion.loader import DocumentLoader
    loader = DocumentLoader()
    all_documents_for_retrieval = []
    for root, _, files in os.walk("data"):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            text_chunks, multimodal_docs = loader.load_and_split(file_path)
            all_documents_for_retrieval.extend(text_chunks)
            all_documents_for_retrieval.extend(multimodal_docs)

    # Initialize RetrievalPipeline
    retrieval_pipeline = RetrievalPipeline(
        documents=all_documents_for_retrieval,
        retrieval_k=5,
        rerank_top_n=2
    )

    query = "What are the applications of neural networks in AI?"
    results = retrieval_pipeline.retrieve_and_rerank(query)

    print(f"Final Reranked Results for '{query}':")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:150]}...")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}, Type: {doc.metadata.get('element_type', 'text')}")

    query_graph = "Show me the market trends graph"
    results_graph = retrieval_pipeline.retrieve_and_rerank(query_graph)
    print(f"Final Reranked Results for '{query_graph}':")
    for i, doc in enumerate(results_graph):
        print(f"Result {i+1}: {doc.page_content[:150]}...")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}, Type: {doc.metadata.get('element_type', 'text')}")
