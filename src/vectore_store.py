import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Dict

class FaissVectorStore():
    def __init__(self, persist_directory : str = 'faiss_store'):
        
        self.persist_dir = persist_directory
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata: List[Dict] = []
        print(f"[INFO] FAISS vector store initialized at {persist_directory} from Vectorstore.py")
    

    def add_embeddings(self, embeddings: np.ndarray, metadata:List[Dict]):
        embeddings = embeddings.astype("float32")
        print('EMBEDDINGS PRINTED BELOW')
        print(embeddings)
        dim = embeddings.shape[1]

        if self.index == None:
            self.index = faiss.IndexFlatL2(dim)
            print(f"[INFO] Created FAISS IndexFlatL2 with dimension {dim} from Vectorstore.py")
        
        self.index.add(embeddings)

        self.metadata.extend(metadata)

        print(f"[INFO] Added {len(embeddings)} vectors to FAISS from Vectorstore.py")
    

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "index.faiss")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index and metadata from Vectorstore.py")


    def load(self):
        faiss_path = os.path.join(self.persist_dir, "index.faiss")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index =  faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir} from Vectorstore.py")


    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.astype("float32")

        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append({
                "index": idx,
                "distance": float(dist),
                "metadata": self.metadata[idx]
            })
        return results