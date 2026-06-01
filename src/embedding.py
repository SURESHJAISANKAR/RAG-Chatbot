from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from data_loader import load_all_docs
from typing import List, Any
from fastembed import TextEmbedding
import numpy as np 

class Embeddding_pipeline:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size : int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name} from Embedding.py")
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks from Embedding.py")
        return chunks

    def embed_chunks(self, chunks : list[Any])-> np.ndarray: 
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks... from Embedding.py")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape} from Embedding.py")
        metadata = []
        for chunk in chunks:
            metadata.append({
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source")
        })
        return embeddings, metadata
