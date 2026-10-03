import streamlit as st
import os
from google.colab import userdata
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Page configurations
st.set_page_config(page_title="College Enquiry Assistant", page_icon="🎓")
st.title("🎓 Smart College Enquiry Bot")
st.subheader("Ask anything about B.Tech admissions, fees, or placements!")

# 2. Re-connect to our Chroma Vector Database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db_fresh", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# 3. Text box setup for users to type their queries
user_query = st.text_input("Enter your question here:")

if user_query:
    with st.spinner("Searching official records..."):
        # Retrieve context from database matching user query
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
        
        # Connect to Groq production model using our safe secret key
        os.environ["GROQ_API_KEY"] = userdata.get('Clgbot')
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
        response = llm.invoke(system_prompt)
        
        # Display the output directly on the web browser interface page
        st.write("### 🤖 Chatbot Response:")
        st.success(response.content)
