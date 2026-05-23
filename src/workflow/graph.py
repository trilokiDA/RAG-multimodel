import operator
import os
from typing import List, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.retrieval.pipeline import RetrievalPipeline
from dotenv import load_dotenv

load_dotenv() # Load environment variables

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question
        documents: List of retrieved documents
        generation: The LLM generation
    """
    question: str
    documents: Annotated[List[Document], operator.add]
    generation: str

def retrieve(state: GraphState, retrieval_pipeline: RetrievalPipeline):
    """
    Retrieve documents from the hybrid retrieval pipeline.
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retrieval_pipeline.retrieve_and_rerank(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """
    Generate a well-structured answer using RAG.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Enhanced Prompt for structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert financial and data analyst assistant. "
            "Use the provided context to answer the user's question with high precision. "
            "\n\nFOLLOW THESE FORMATTING RULES:"
            "\n1. Use professional Markdown formatting."
            "\n2. Use ## Headers to separate different sections of your analysis."
            "\n3. If the data contains numbers or comparisons, use Markdown TABLES to present them."
            "\n4. Use **bold text** for key findings or specific data points."
            "\n5. Use bullet points for lists."
            "\n6. If the answer is not in the context, state clearly that the information is missing."
            "\n7. Keep the tone professional and objective."
        )),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])

    # Groq LLM
    # Note: Ensure the model name is correct for your Groq setup (e.g., 'llama3-70b-8192')
    llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

    # Chain
    rag_chain = prompt | llm 

    generation = rag_chain.invoke({"context": documents, "question": question})
    
    # Check for multimodal documents and append info
    image_paths = [doc.metadata.get("image_path") for doc in documents if doc.metadata.get("is_multimodal") and doc.metadata.get("image_path")]
    
    final_response = generation.content
    if image_paths:
        final_response += "\n\n## 🖼️ Relevant Visuals Found"
        for path in set(image_paths):
            final_response += f"\n- **Source Image:** `{path}`"

    return {"documents": documents, "question": question, "generation": final_response}
