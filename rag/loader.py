

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(file_path: str):
    
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        
        loader = PyPDFLoader(str(path))

    elif suffix in (".txt", ".md"):
        
        loader = TextLoader(str(path), encoding="utf-8")

    else:
        
        raise ValueError(
            f"Unsupported file type: '{suffix}'. Supported types: .pdf, .txt, .md"
        )

    documents = loader.load()

   
    for doc in documents:
        doc.metadata["source"] = path.name

    return documents


def load_all_documents(file_paths: list[str]):
    
    all_documents = []
    for file_path in file_paths:
        docs = load_document(file_path)
        all_documents.extend(docs)
    return all_documents
