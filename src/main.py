from data_loader import load_all_docs
from embedding import Embeddding_pipeline
from vectore_store import FaissVectorStore

class RAGPipeline:
    def __init__(self, data_path=f'D:\RAG-Chatbot\data'):
        self.data_path = data_path
        print(f'[INFO] Data Path Initialised at {data_path} from Main.py')
        self.pipeline = Embeddding_pipeline()
        self.vector_store = FaissVectorStore()

    def build(self):
        docs = load_all_docs(self.data_path)
        print(f'[INFO] Docs Ready for Chunking {len(docs)}')
        chunks = self.pipeline.chunk_documents(docs)
        embeddings, metadata = self.pipeline.embed_chunks(chunks)
        self.vector_store.add_embeddings(embeddings, metadata)
        self.vector_store.save()
        print("[INFO] Vector store built successfully ✅ from Main.py")

    def load(self):
        self.vector_store.load()


    def query(self, query_text: str, top_k=3):
        query_embedding = self.pipeline.model.encode([query_text]).astype("float32")

        results = self.vector_store.search(query_embedding, top_k)

        context = "\n".join([r["metadata"]["text"] for r in results])

        return context


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.build()
