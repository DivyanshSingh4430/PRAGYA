import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Page config and design
st.set_page_config(page_title="Pragya - Class 10 Science Chatbot")
st.markdown("<h1 style='text-align: center; color: #4a90e2;'>🧠 Pragya: NCERT Class 10 Science Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>CREATED BY DIVYANSH SINGH</p>", unsafe_allow_html=True)

# Load and split the document
loader = PyPDFLoader("ncert_class10_science.pdf")
pages = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Embed & create retriever
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Create QA chain
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# UI input
query = st.text_input("❓ Ask a question from the NCERT Science Book")

if query:
    result = qa_chain.run(query)
    st.write("🟦 **Answer:**", result)
