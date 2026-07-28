import streamlit as st
import tempfile
import os
import time

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_community.retrievers import BM25Retriever

# FIXED IMPORT: Using langchain_classic for the EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever

st.set_page_config(
    page_title="Omni-Data Hybrid Engine", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS for a professional, tech-expert UI
st.markdown("""
<style>
    /* Hide standard Streamlit header and footer for a clean app look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Professional Dark Theme Adjustments */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Floating, sleek chat input box */
    .stChatInput { 
        border-radius: 12px !important; 
        border: 1px solid #333 !important;
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.4) !important; 
        background-color: #161B22 !important;
    }

    /* Sidebar styling: subtle borders and tech-focused look */
    [data-testid="stSidebar"] { 
        border-right: 1px solid #30363D; 
        background-color: #010409;
    }

    /* Glowing dropzone for file uploads */
    [data-testid="stFileUploadDropzone"] { 
        border: 2px dashed #58A6FF; 
        border-radius: 12px; 
        background-color: rgba(88, 166, 255, 0.05); 
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(88, 166, 255, 0.15);
        border: 2px dashed #79C0FF;
    }

    /* Styling for the Source Verification expanders */
    .streamlit-expanderHeader { 
        background-color: #161B22 !important; 
        border-radius: 8px !important; 
        font-size: 0.9rem !important;
        color: #8B949E !important;
        border: 1px solid #30363D !important;
    }
    
    /* Custom Scrollbar for a premium feel */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0E1117; 
    }
    ::-webkit-scrollbar-thumb {
        background: #30363D; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #58A6FF; 
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚡ Omni-Data Engine")
    st.caption("Hybrid RAG Architecture (BM25 + Vector)")
    st.divider()
    
    st.markdown("#### 📂 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Unstructured Data", 
        type=["pdf", "docx", "xlsx", "txt"],
        help="Supports PDF, Word, Excel, and Text files via Unstructured OCR."
    )
    
    st.divider()
    
    st.markdown("#### ⚙️ System Controls")
    if st.button("🧹 Flush Memory & Reset", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption("Model: Llama-3.2-3B")
    st.caption("Embeddings: nomic-embed-text")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Omni-Data Hybrid Engine. Please upload a dataset in the sidebar to begin our session.", "sources": []}
    ]

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Context (RRF)"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {idx+1}:**\n> _{src['content']}_")

def stream_response(text):
    """Simulates a streaming effect for the AI response."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) # Adjust speed of typing here

if uploaded_file:
    @st.cache_resource(show_spinner="⚙️ Ingesting & Indexing Data (BM25 + Chroma)...")
    def process_document(file_bytes, filename):
        ext = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            tf.write(file_bytes)
            file_path = tf.name

        try:
            # Leverage Unstructured for multi-format and OCR capabilities
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()
        except Exception as e:
            # EXPLICIT ERROR CAPTURE: Reveals missing packages like 'poppler' or 'tesseract'
            return None, f"Parsing Error: {str(e)}"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        os.remove(file_path)

        if not chunks:
            return None, "No readable text found in the document."

        # 1. Semantic Vector Search (ChromaDB)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(chunks, embeddings)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 2. Exact Keyword Search (BM25)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3

        # 3. Hybrid Search Engine (Ensemble via Reciprocal Rank Fusion)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5] # 50% keyword, 50% semantic
        )
        
        return ensemble_retriever, None

    # Execute processing
    ensemble_retriever, error = process_document(uploaded_file.getvalue(), uploaded_file.name)
    
    if error:
        st.sidebar.error(f"**Ingestion Failed:**\n{error}")
        st.sidebar.info("💡 Tip: If it says 'tesseract' or 'poppler' is missing, install them on your Windows machine.")
        st.stop()
        
    st.sidebar.success(f"🟢 Active Index: **{uploaded_file.name}**")

    if query := st.chat_input("Query the hybrid database..."):
        # Append and display user query
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Querying Hybrid Engine..."):
                llm = OllamaLLM(model="llama3.2")
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Fetch optimal chunks
                matching_docs = ensemble_retriever.invoke(query)
                
                # Advanced System Prompt for better engagement
                interactive_query = (
                    f"Context: Answer the following question thoroughly based strictly on the provided documents.\n"
                    f"Question: {query}\n\n"
                    f"Format your response beautifully using Markdown (bolding, bullet points). "
                    f"At the very end of your response, add a section called '### 💡 Suggested Follow-up' "
                    f"and provide one highly analytical follow-up question the user could ask."
                )
                
                # Invoke LLM
                response = chain.invoke({"input_documents": matching_docs, "question": interactive_query})
                answer = response["output_text"]
                
                # Stream the output for engaging UX
                st.write_stream(stream_response(answer))
                
                # Format sources for transparency
                sources = [{"content": doc.page_content[:300].replace("\n", " ") + "..."} for doc in matching_docs]
                
                if sources:
                    with st.expander("🔍 View Retrieved Context (RRF)"):
                        for idx, src in enumerate(sources):
                            st.markdown(f"**Chunk {idx+1}:**\n> _{src['content']}_")

        # Save to memory
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})