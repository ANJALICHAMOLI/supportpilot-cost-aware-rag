
def get_retriever(vector_store, k: int = 4):

    return vector_store.as_retriever(search_type="similarity",search_kwargs={"k": k},)


def retrieve_relevant_chunks(vector_store, query: str, k: int = 4):

    retriever = get_retriever(vector_store, k=k)

    results = retriever.invoke(query)
    return results
