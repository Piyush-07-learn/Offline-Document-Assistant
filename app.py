import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains.question_answering import load_qa_chain

st.title("📄 Local PDF Assistant")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
query = st.text_input("Ask a question about your PDF:")

if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    with st.spinner("Analyzing PDF... this may take a moment."):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(chunks, embeddings)

        llm = OllamaLLM(model="llama3.2")
        chain = load_qa_chain(llm, chain_type="stuff")
        matching_docs = vector_store.similarity_search(query)
        
        response = chain.invoke({"input_documents": matching_docs, "question": query})
        
        st.write("### Answer:")
        st.write(response["output_text"])

    os.remove(file_path)