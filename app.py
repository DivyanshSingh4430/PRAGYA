import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

st.set_page_config(page_title="Pragya - Class 10 Science Chatbot")

st.markdown("<h1 style='text-align: center; color: #4a90e2;'>🧠 Pragya: NCERT Class 10 Science Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>CREATED BY DIVYANSH SINGH</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload your NCERT Class 10 Science PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Embed
    embeddings = OpenAIEmbeddings()
    vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)

    # QA Chain
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")

    query = st.text_input("❓ Ask a question from the NCERT Science Book")

    if query:
        docs = vectorstore.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.write("📘 Answer:", response)
