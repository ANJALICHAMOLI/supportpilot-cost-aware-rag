

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        
    )

    chunks = splitter.split_documents(documents)
    return chunks
