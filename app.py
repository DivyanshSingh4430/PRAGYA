import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator

# Set OpenAI key from secrets
openai_key = os.getenv("OPENAI_API_KEY")

# Title & styling
st.set_page_config(page_title="Pragya - Class 10 Science Chatbot")
st.markdown("<h1 style='text-align: center; color: #4a90e2;'>🧠 Pragya: NCERT Class 10 Science Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>CREATED BY DIVYANSH SINGH</p>", unsafe_allow_html=True)

# Load PDF
loader = PyPDFLoader("ncert_class10_science.pdf")
index = VectorstoreIndexCreator().from_loaders([loader])

# Input question
query = st.text_input("❓ Ask a question from the NCERT Science Book")

# Handle answer
if query:
    answer = index.query(query, llm=OpenAI(openai_api_key=openai_key))
    st.write("📘 **Answer:**", answer)
