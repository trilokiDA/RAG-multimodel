import os
import hashlib
import json
from typing import List, Dict
from src.ingestion.loader import DocumentLoader
from src.ingestion.vector_store import VectorStoreManager
from langchain_core.documents import Document

class IngestionPipeline:
    def __init__(self, data_dir: str = "data", chroma_persist_dir: str = "./chroma_db"):
        self.data_dir = data_dir
        self.document_loader = DocumentLoader()
        self.vector_store_manager = VectorStoreManager(persist_directory=chroma_persist_dir)
        self.metadata_file = os.path.join(chroma_persist_dir, "ingestion_metadata.json")

    def _get_file_hash(self, file_path: str) -> str:
        """Calculates the MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_metadata(self) -> Dict[str, str]:
        """Loads the ingestion metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f).get("files", {})
            except (json.JSONDecodeError, KeyError):
                return {}
        return {}

    def _save_metadata(self, metadata: Dict[str, str]) -> None:
        """Saves the ingestion metadata."""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump({"files": metadata}, f, indent=4)

    def run(self) -> None:
        """
        Runs the ingestion pipeline incrementally:
        1. Checks for changes in the data directory using file hashes.
        2. Only processes new or modified files.
        3. Updates the vector store and metadata.
        """
        print(f"Starting ingestion from {self.data_dir}...")
        current_metadata = self._load_metadata()
        new_metadata = {}
        all_new_chunks: List[Document] = []
        
        # Track if any file was processed or if any file was skipped but still exists
        files_found_count = 0

        for root, _, files in os.walk(self.data_dir):
            for file_name in files:
                files_found_count += 1
                file_path = os.path.join(root, file_name)
                file_hash = self._get_file_hash(file_path)
                
                if file_path in current_metadata and current_metadata[file_path] == file_hash:
                    print(f"Skipping {file_path} (already indexed).")
                    new_metadata[file_path] = file_hash
                    continue
                
                print(f"Processing {file_path}...")
                try:
                    text_chunks, multimodal_docs = self.document_loader.load_and_split(file_path)
                    print(f"  Extracted {len(text_chunks)} text chunks and {len(multimodal_docs)} images.")
                    all_new_chunks.extend(text_chunks)
                    all_new_chunks.extend(multimodal_docs)
                    new_metadata[file_path] = file_hash
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    # Keep old hash if processing failed so we retry next time
                    if file_path in current_metadata:
                        new_metadata[file_path] = current_metadata[file_path]

        if all_new_chunks:
            print(f"Adding {len(all_new_chunks)} total new chunks to vector store...")
            self.vector_store_manager.add_documents(all_new_chunks)
            self._save_metadata(new_metadata)
            print(f"Ingestion complete. Added {len(all_new_chunks)} new chunks.")
        elif files_found_count > 0:
            # Check if we should update metadata even if no new chunks (e.g., all files skipped)
            # We should save if the set of files has changed (e.g., some were deleted)
            if new_metadata != current_metadata:
                self._save_metadata(new_metadata)
            print("No new or modified documents found to process.")
        else:
            print("No documents found in data directory.")

if __name__ == "__main__":
    # Example usage:
    # Create a dummy file for testing
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/example.txt", "w") as f:
        f.write("This is an example text document. It talks about AI and machine learning.")
    with open("data/example.pdf", "w") as f: # This won't be a valid PDF, but for path simulation
        f.write("This is a dummy PDF content. Imagine it has graphs.")

    pipeline = IngestionPipeline()
    pipeline.run()

    # To test loading the vector store:
    # loaded_vector_store = pipeline.vector_store_manager.load_vector_store()
    # print(f"Loaded vector store with {loaded_vector_store._collection.count()} documents.")
    # results = loaded_vector_store.similarity_search("What is AI?")
    # print(results)
