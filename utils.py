from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader

def load_and_split(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    else:
        loader = TextLoader(file_path)
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
