# RAG-Powered College Enquiry Chatbot System 🎓

An AI-powered conversational agent built using **Retrieval-Augmented Generation (RAG)** to automate student admissions and campus support queries accurately without hallucinations.

## 🚀 Key Features
- **Zero Hallucination:** Answers are strictly anchored to verified college documents.
- **Context Retrieval:** Uses vector embeddings to scan institutional text files.
- **LLM Engine:** Powered by GPT-OSS-20B running via Groq API framework.

## 🛠️ Architecture & Tech Stack
- **Orchestration:** LangChain Framework
- **Vector Database:** ChromaDB
- **Embedding Model:** HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Model:** GPT-OSS-20B (`openai/gpt-oss-20b`)

## 📋 Project Execution Setup
1. Clone the project pipeline text structures.
2. Run data ingestion setup: `python ingest.py`
3. Run target query interface pipeline: `python app.py`
