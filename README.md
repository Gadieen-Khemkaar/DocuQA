📘 Document QA Bot
A Question-Answering (QA) Bot built with LangChain that lets you query documents (PDFs, text) using LLMs such as Groq (LLaMA-3/Mixtral) or Gemini.

Ask questions to your documents and get instant, accurate answers with source references.

🚀 Features

📄 Load PDFs or text files

🔢 Free, local embeddings via HuggingFace (all-MiniLM-L6-v2)

⚡ Fast LLM inference with Groq or Gemini

🧠 Returns answers + source references

✂️ Handles long documents by chunking with overlap for context preservation

🔄 Modular design: switch between LLMs or embeddings easily

⚙️ Installation
# Core dependencies
pip install langchain langchain-community faiss-cpu pypdf sentence-transformers

# Optional: Groq support
pip install langchain-groq

# Optional: Gemini (Google Generative AI) support
pip install langchain-google-genai

🛠️ Usage Example
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # or ChatGoogleGenerativeAI

# 1️⃣ Load document
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# 2️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Store in vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# 5️⃣ Setup retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 6️⃣ Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# 7️⃣ Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 8️⃣ Ask a question
query = "What are the main findings of the document?"
response = qa_chain.invoke({"query": query})

print("Answer:", response["result"])
print("Sources:", [doc.metadata for doc in response["source_documents"]])

🔎 Code Breakdown
Component	Description
Document Loader	PyPDFLoader extracts text & metadata from PDFs
Text Splitter	RecursiveCharacterTextSplitter breaks text into chunks with overlap
Embeddings	HuggingFaceEmbeddings converts text into numerical vectors
Vector Store	FAISS stores embeddings for fast similarity search
Retriever	Fetches top-k relevant chunks for a query
LLM	ChatGroq or ChatGoogleGenerativeAI generates answers
QA Chain	RetrievalQA connects retriever + LLM and returns answer + sources
🔄 Workflow Diagram
User Query
     │
     ▼
  Retriever (FAISS)
     │
     ▼
Top-k Relevant Chunks
     │
     ▼
     LLM (Groq / Gemini)
     │
     ▼
Answer + Source Documents

💡 Example Output
Query: "What are the main findings of the document?"

Answer: The document concludes that AI improves diagnostic accuracy, reduces operational costs, 
and enhances patient outcomes, but faces challenges in data privacy and regulatory approval.

Sources: [{'page': 5}, {'page': 12}]

⚠️ Notes

Switching LLMs: Replace ChatGroq with ChatGoogleGenerativeAI for Gemini.

Embeddings: You can use other HuggingFace models if needed.

Chunk size & overlap: Adjust depending on document size and LLM context window.

FAISS: Can persist to disk for large datasets using vectorstore.save_local()

