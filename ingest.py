import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_database():
    if os.path.exists("chroma_db_fresh"):
        shutil.rmtree("chroma_db_fresh")
        
    if not os.path.exists("college_faq.txt"):
        print("Error: college_faq.txt missing!")
        return
        
    loader = TextLoader("college_faq.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    docs = [Document(page_content=c.page_content) for c in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db_fresh")
    print("Vector database built successfully!")

if __name__ == "__main__":
    build_database()
