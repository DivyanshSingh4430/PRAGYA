import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Page Setup
st.set_page_config(page_title="Pragya - Class 10 Science Chatbot")
st.markdown("<h1 style='text-align: center; color: #4a90e2;'>🧠 Pragya: NCERT Class 10 Science Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>CREATED BY DIVYANSH SINGH</p>", unsafe_allow_html=True)

# Load and process the PDF
loader = PyPDFLoader("ncert_class10_science.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Load API key from Streamlit secrets
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(texts, embedding=embeddings)

# LLM and chain
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

# Input
query = st.text_input("❓ Ask a question from the NCERT Science Book")
if query:
    docs = vectorstore.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    st.write("📘 **Answer:**", response)
