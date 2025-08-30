from langchain.retrievers import BM25Retriever

def build_bm25_retriever(chunks):
    texts = [doc.page_content for doc in chunks]
    return BM25Retriever.from_texts(texts)
