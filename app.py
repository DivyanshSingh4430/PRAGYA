import os
import streamlit as st
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI

loader = PyPDFLoader("ncert_class10_science.pdf")
index = VectorstoreIndexCreator().from_loaders([loader])

query = st.text_input("❓ Ask a question from the NCERT Science Book")
if query:
    response = index.query(query, llm=OpenAI(openai_api_key=openai_key))
    st.write("📘 **Answer:**", response)


# API Key from Streamlit Secrets
openai_key = os.getenv("OPENAI_API_KEY")

# UI Design
st.set_page_config(page_title="Pragya - Class 10 Science Chatbot")
st.markdown("<h1 style='text-align: center; color: #4a90e2;'>🧠 Pragya: NCERT Class 10 Science Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>CREATED BY DIVYANSH SINGH</p>", unsafe_allow_html=True)

# Load the NCERT PDF
loader = PyPDFLoader("ncert_class10_science.pdf")
index = VectorstoreIndexCreator().from_loaders([loader])

# Input and Answer
query = st.text_input("❓ Ask a question from the NCERT Science Book")
if query:
    response = index.query(query, llm=OpenAI(openai_api_key=openai_key))
    st.write("📘 **Answer:**", response)
