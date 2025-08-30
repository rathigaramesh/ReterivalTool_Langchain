from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY

def run_chain_mode(query, retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    # Use the correct input format for RetrievalQA
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = [doc.page_content[:500] for doc in result["source_documents"]]
    return answer, sources