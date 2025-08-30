from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_vector_retriever(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
