import os
import fitz  # PyMuPDF
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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

    def _extract_images_from_pdf(self, file_path: str) -> List[Document]:
        """
        Extracts images from a PDF using PyMuPDF and returns a list of Documents 
        containing the image paths.
        """
        multimodal_docs = []
        doc_name = os.path.basename(file_path).split('.')[0]
        
        try:
            pdf_document = fitz.open(file_path)
            for page_index in range(len(pdf_document)):
                page = pdf_document[page_index]
                image_list = page.get_images(full=True)
                
                for image_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image_filename = f"{doc_name}_pg{page_index+1}_img{image_index+1}.{image_ext}"
                    image_path = os.path.join(self.image_output_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    description = (
                        f"This is an image extracted from {file_path} on page {page_index + 1}. "
                        f"It is stored at {image_path}. This image may contain relevant charts, "
                        f"tables, or diagrams mentioned on this page."
                    )
                    
                    multimodal_docs.append(Document(
                        page_content=description,
                        metadata={
                            "source": file_path,
                            "page": page_index + 1,
                            "element_type": "image",
                            "image_path": image_path,
                            "is_multimodal": True
                        }
                    ))
            pdf_document.close()
        except Exception as e:
            print(f"Error extracting images with PyMuPDF: {e}")
            
        return multimodal_docs

    def load_and_split(self, file_path: str) -> Tuple[List[Document], List[Document]]:
        """
        Loads a file using PyPDFLoader for text and PyMuPDF for image extraction.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.lower().endswith(".pdf"):
            print(f"  Extracting text and images from {file_path}...")
            
            # 1. Extract Text
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text_chunks = self.text_splitter.split_documents(pages)
            
            # 2. Extract Images
            multimodal_docs = self._extract_images_from_pdf(file_path)
            
            print(f"  Processed {len(text_chunks)} text chunks and {len(multimodal_docs)} images.")
            return text_chunks, multimodal_docs
        else:
            print(f"  Warning: Only PDF is fully supported. Skipping {file_path}")
            return [], []
