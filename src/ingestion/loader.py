import os
from typing import List, Tuple
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from unstructured.partition.auto import partition
from unstructured.documents.elements import Image, Text, Title, NarrativeText, ListItem

from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, image_output_dir: str = "extracted_images"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.image_output_dir = image_output_dir
        if not os.path.exists(self.image_output_dir):
            os.makedirs(self.image_output_dir)

    def load_and_split(self, file_path: str) -> Tuple[List[Document], List[Document]]:
        """
        Loads a file using PyPDFLoader (fallback from Unstructured due to environment issues).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.lower().endswith(".pdf"):
            print(f"  Using PyPDFLoader for {file_path}...")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Split pages into chunks
            text_chunks = self.text_splitter.split_documents(pages)
            
            # PyPDFLoader doesn't extract images easily. 
            # We'll return an empty list for multimodal docs for now to allow system to run.
            return text_chunks, []
        else:
            # For non-PDF files, we might still need a generic loader or unstructured if it works for text
            print(f"  Warning: Only PDF is fully supported with fallback loader. Skipping {file_path}")
            return [], []
