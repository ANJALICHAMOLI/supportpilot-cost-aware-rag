

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PERSIST_DIR = "./chroma_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks, persist_directory: str = CHROMA_PERSIST_DIR):

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="supportpilot_docs",
    )
    return vector_store


def load_vector_store(persist_directory: str = CHROMA_PERSIST_DIR):
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name="supportpilot_docs",
    )
    return vector_store


def vector_store_exists(persist_directory: str = CHROMA_PERSIST_DIR) -> bool:
    return os.path.isdir(persist_directory) and len(os.listdir(persist_directory)) > 0
