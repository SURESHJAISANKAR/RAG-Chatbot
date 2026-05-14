from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from typing import List, Any, Dict

def load_all_docs(data_dir: str) -> List[Any]:
    directory_path = Path(data_dir).resolve()
    documents = []
    pdf_files = list(directory_path.glob('**/*.pdf'))
    for pdf in pdf_files:
        loader = PyMuPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents