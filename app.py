import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def run_chatbot(user_query):
    if "GROQ_API_KEY" not in os.environ:
        return "Error: GROQ_API_KEY environment variable not set."

    # Force load embeddings cleanly
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db_fresh", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    relevant_docs = retriever.invoke(user_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    system_prompt = f"""
    You are an official college admissions assistant. Answer the user's question using ONLY the provided text context below. 
    If you do not know the answer based on the context, say 'I cannot find that in the official prospectus. Please email admissions@college.edu.'

    Context:
    {context}

    Question: {user_query}
    Answer:
    """

    # Using the production-ready stable model to eliminate the 404 error
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
    response = llm.invoke(system_prompt)
    return response.content

if __name__ == "__main__":
    print(run_chatbot("What is the fee for B.Tech courses?"))
