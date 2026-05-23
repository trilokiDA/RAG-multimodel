# RAG Pipeline with LangGraph, Groq, and Multimodal Support

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to process various document types, perform hybrid retrieval, rerank results, and generate answers using Groq's LLMs, all orchestrated by LangGraph. It also includes preliminary support for multimodal content (like graphs in PDFs).

## Features

*   **Multimodal Ingestion:** Handles PDF, MS Word, PPT, and plain text documents using `unstructured`. Includes a placeholder for extracting and describing visual elements (graphs, charts) from PDFs using a future vision-capable LLM.
*   **Hybrid Retrieval:** Combines sparse retrieval (BM25) and dense retrieval (vector embeddings via ChromaDB) for comprehensive document fetching.
*   **Document Reranking:** Utilizes a BGE-Reranker model to improve the relevance of retrieved documents.

*   **LangGraph Orchestration:** Manages the workflow from retrieval to generation using a state-based graph.
*   **Groq LLM Integration:** Uses Groq's high-performance language models (e.g., `mixtral-8x7b-32768`) for generating answers.

## Project Structure

```
rag-pipeline/
├── data/               # Raw document storage
├── src/
│   ├── ingestion/      # Document loaders, chunking, and vector store management
│   ├── retrieval/      # Hybrid retriever, reranking, and caching
│   ├── workflow/       # LangGraph state machine definitions and nodes
│   └── utils/          # Utility functions (e.g., for multimodal processing) - currently integrated
├── main.py             # Main entry point for the RAG application
└── requirements.txt    # Python dependencies

```

## Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    # If this were a real repository
    # git clone <repository_url>
    # cd rag-pipeline
    ```
    (Assuming you are already in the `D:\Project\Rag-all-features` directory)

2.  **Install Dependencies:**
    Ensure you have Python 3.9+ installed. Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

**Groq API Key:**

Create a file named `.env` in the root directory (`D:\Project\Rag-all-features\.env`) and add your Groq API key:

```
GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
```

Replace `"YOUR_GROQ_API_KEY_HERE"` with your actual API key obtained from the [Groq console](https://console.groq.com/keys).

## Usage

1.  **Place Your Documents:**
    Put all the documents (PDF, DOCX, PPTX, TXT) you want the RAG system to process into the `data/` directory.

    Example:
    ```
    rag-pipeline/
    ├── data/
    │   ├── my_report.pdf
    │   ├── company_policy.docx
    │   └── meeting_notes.txt
    └── ...
    ```

2.  **Run the Application:**
    Execute the `main.py` script from the project root:

    ```bash
    python main.py
    ```

    Upon running, the application will:
    *   Initialize the ingestion pipeline, processing documents from the `data/` directory and creating/updating the `chroma_db` vector store.
    *   Set up the hybrid retrieval and reranking components.
    *   Compile the LangGraph workflow.

    You will then be prompted to ask questions:

    ```
    ---Setting up Ingestion Pipeline---
    ... (ingestion logs) ...
    ---Setting up Retrieval Pipeline---
    ... (retrieval setup logs) ...
    ---LangGraph compiled successfully---

    ---RAG System Ready!---
    Type 'exit' to quit.
    Ask a question: What is the main topic of the company policy document?
    ---
    (LLM generation output)
    ---
    Ask a question:
    ```

## Future Enhancements

*   **Actual Multimodal LLM Integration:** Replace the current placeholder for multimodal element processing with a real vision-capable LLM to genuinely understand and describe graphs and images within documents.
*   **Error Handling and Robustness:** Enhance error handling for document processing failures and API calls.
*   **Advanced LangGraph Flows:** Implement more complex LangGraph workflows, such as self-correction, query expansion, or routing based on query intent.
*   **User Interface:** Develop a simple web or command-line interface for easier interaction.
*   **Configuration Management:** Externalize more configurations (e.g., chunk sizes, model names) to a dedicated config file.
