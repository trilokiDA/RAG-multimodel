from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.reranker_model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        """
        Reranks a list of documents based on their relevance to the query.
        """
        if not documents:
            return []

        # Prepare pairs for reranking model: (query, document_content)
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores from the reranker model
        scores = self.reranker_model.predict(pairs)

        # Apply a boost for multimodal documents to help them surface
        for i, doc in enumerate(documents):
            if doc.metadata.get("is_multimodal"):
                scores[i] += 0.5 # Small boost to increase chance of inclusion

        # Combine documents with their scores and sort in descending order
        scored_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        # Return top_n reranked documents
        reranked_docs = [doc for doc, score in scored_documents[:top_n]]
        
        print(f"Reranked {len(documents)} documents, returning top {top_n}.")
        return reranked_docs

if __name__ == "__main__":
    print("--- Testing DocumentReranker ---")
    reranker = DocumentReranker()

    # Create some dummy documents
    dummy_documents = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"id": 1}),
        Document(page_content="Artificial intelligence is a rapidly evolving field.", metadata={"id": 2}),
        Document(page_content="Machine learning is a subset of AI.", metadata={"id": 3}),
        Document(page_content="Quantum physics deals with subatomic particles.", metadata={"id": 4}),
        Document(page_content="AI has many applications in various industries.", metadata={"id": 5}),
    ]

    query = "applications of AI"
    print(f"Original documents (first 50 chars):")
    for i, doc in enumerate(dummy_documents):
        print(f"{i+1}. {doc.page_content[:50]}...")

    print(f"Query: '{query}'")
    reranked_results = reranker.rerank(query, dummy_documents, top_n=2)

    print(f"Reranked documents (top 2):")
    for i, doc in enumerate(reranked_results):
        print(f"{i+1}. {doc.page_content[:50]}...")
        print(f"   Metadata ID: {doc.metadata.get('id')}")

    print("--- Testing with no documents ---")
    reranked_empty = reranker.rerank("test query", [], top_n=1)
    print(f"Reranked empty list: {reranked_empty}")
