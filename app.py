import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="DocuQA", layout="wide")

st.title("📄 DocuQA: Chat with Your PDF Agent")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    try:
        st.info("Saving uploaded PDF...")
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        st.success("PDF saved. Loading document...")
        loader = PyPDFLoader("temp_uploaded.pdf")
        documents = loader.load()
        st.success(f"Loaded {len(documents)} document(s). Splitting text...")


    # Read Groq API key from environment
    api_key = os.getenv("GROQ_API_KEY")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        st.success(f"Split into {len(docs)} chunks. Creating embeddings...")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.success("Embeddings created. Setting up retriever and LLM...")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        st.success("PDF loaded and processed! Start chatting below.")

        # Display chat history
        for i, (user, agent) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {user}")
            st.markdown(f"**Agent:** {agent}")

        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your question and press Enter:", key="user_input")
            submit = st.form_submit_button("Send")
            if submit and user_input:
                st.info("Agent is thinking...")
                response = qa_chain.invoke({"query": user_input})
                answer = response["result"]
                st.session_state.chat_history.append((user_input, answer))
                st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF to get started.")
