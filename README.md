# RAG Pipeline with LangGraph, Groq, and Multimodal Support

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to process various document types, perform hybrid retrieval, rerank results, and generate answers using Groq's LLMs, all orchestrated by LangGraph. It provides advanced support for retrieving and rendering visual content (graphs, charts) extracted from PDF documents.

## Features

*   **Multimodal Ingestion & Enrichment:** Extracts images from PDFs and enriches them with page-level text summaries to ensure accurate semantic retrieval.
*   **Structured Multimodal Response:** The backend returns text generations and associated image paths separately, designed for seamless integration with frontend UIs.
*   **Hybrid Retrieval:** Combines sparse retrieval (BM25) and dense retrieval (vector embeddings via ChromaDB) for comprehensive document fetching.
*   **Multimodal-Aware Reranking:** Uses a BGE-Reranker model with built-in boosting for multimodal documents, ensuring that relevant images are preserved in top-N search results.
*   **LangGraph Orchestration:** Manages the workflow from retrieval to generation using a state-based graph.
*   **Groq LLM Integration:** Uses Groq's high-performance language models for generating answers.

## Project Structure

```
rag-pipeline/
├── data/               # Raw document storage
├── extracted_images/   # Directory for PDF-extracted images
├── src/
│   ├── ingestion/      # Document loaders (PDF text + image extraction), chunking, vector store management
│   ├── retrieval/      # Hybrid retriever, reranker (with multimodal boost), and caching
│   ├── workflow/       # LangGraph state machine definitions, prompt engineering
│   └── utils/          # Utility functions
├── main.py             # Main entry point (prints text and image paths separately)
└── requirements.txt    # Python dependencies
```

## Setup and Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

**Groq API Key:**
Create a file named `.env` in the root directory and add your key:
```
GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
```

## Usage

1.  **Place Your Documents:** Put PDF documents containing images into the `data/` directory.
2.  **Run the Streamlit UI:**
    For an interactive web-based UI that renders both the generated text and the images, run:
    ```bash
    streamlit run app.py
    ```
3.  **Run the CLI:**
    Alternatively, for a command-line interface:
    ```bash
    python main.py
    ```
    The CLI will output the textual analysis and list the paths of any relevant images found. You can then map these paths to `<img>` tags if you are building your own custom frontend.

## Performance Note
For optimal multimodal retrieval, ensure you delete the `chroma_db/` folder when upgrading the ingestion logic (e.g., if you change how images are described), forcing the pipeline to re-index documents with the updated, enriched metadata.
