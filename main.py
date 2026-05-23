import os
from typing import List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline
from src.workflow.graph import GraphState, retrieve, generate
from dotenv import load_dotenv

load_dotenv() # Load environment variables

def setup_rag_pipeline(data_dir: str = "data", chroma_persist_dir: str = "./chroma_db"):
    """
    Sets up the ingestion and retrieval pipelines.
    """
    print("---Setting up Ingestion Pipeline---")
    ingestion_pipeline = IngestionPipeline(data_dir=data_dir, chroma_persist_dir=chroma_persist_dir)
    ingestion_pipeline.run()

    print("---Setting up Retrieval Pipeline---")
    # HybridRetriever now handles loading documents from Chroma if 'documents' is empty
    retrieval_pipeline = RetrievalPipeline(
        documents=[], 
        chroma_persist_dir=chroma_persist_dir,
        retrieval_k=10, # Increased from 5 to retrieve more candidates
        rerank_top_n=3 # Example: rerank to top 3
    )
    return retrieval_pipeline

def build_graph(retrieval_pipeline: RetrievalPipeline):
    """
    Builds and compiles the LangGraph.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", lambda state: retrieve(state, retrieval_pipeline)) # Pass retrieval_pipeline
    workflow.add_node("generate", generate)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()
    print("---LangGraph compiled successfully---")
    return app

if __name__ == "__main__":
    # Ensure GROQ_API_KEY is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it in a .env file or as an environment variable.")
        exit(1)

    # Setup pipelines
    rag_retrieval_pipeline = setup_rag_pipeline()
    
    # Build and compile the graph
    app = build_graph(rag_retrieval_pipeline)

    print("RAG System Ready!---")
    print("Type 'exit' to quit.")

    while True:
        question = input("Ask a question: ")
        if question.lower() == 'exit':
            break
        
        # Invoke the graph
        inputs = {"question": question}
        result = app.invoke(inputs)
        
        print("\n---ANSWER---")
        print(result["generation"])
        
        if result.get("image_paths"):
            print("\n---IMAGE PATHS---")
            for path in result["image_paths"]:
                print(path)
        print("\n---")
