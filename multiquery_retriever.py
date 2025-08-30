from langchain.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

def build_multiquery_retriever(vector_retriever, api_key):
    return MultiQueryRetriever.from_llm(
        retriever=vector_retriever,
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    )
