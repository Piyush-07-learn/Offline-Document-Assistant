import os
import shutil
import tempfile
import streamlit as st

# --- LangChain Modern Imports ---
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever   #   for LangChain ≥1.0
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. INITIALIZATION & UI SETUP
# ==========================================

st.set_page_config(page_title="Omni-Data Hybrid RAG", layout="wide", page_icon="⚡")

# Custom CSS inspired by the provided design 
st.markdown("""
<style>
    /* Global Dark Theme & Typography */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #8b949e;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    
    /* Feature Cards */
    .feature-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
    }
    .feature-icon {
        color: #00d2ff;
        font-size: 1.5rem;
        margin-bottom: 10px;
    }
    .feature-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .feature-text {
        font-size: 0.9rem;
        color: #8b949e;
    }
    
    /* Chat & Input Styling */
    .stChatInputContainer > div {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 20px;
    }
    
    /* Source Citation Styling */
    .source-box {
        background-color: #1e242c;
        border-left: 3px solid #00d2ff;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.85rem;
        border-radius: 4px;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SYSTEM DEPENDENCY CHECKS
# ==========================================
def check_dependencies():
    missing = []
    if not shutil.which("tesseract"):
        missing.append("Tesseract OCR (Needed for image/scanned PDF parsing)")
    if not shutil.which("pdftoppm") and not shutil.which("pdftocairo"):
        missing.append("Poppler (Needed for PDF processing)")
    return missing

# ==========================================
# 3. ENGINE CORE FUNCTIONS
# ==========================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Load local Ollama models for embeddings and generation."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.2", temperature=0.1, streaming=True)
    return embeddings, llm

def process_documents(uploaded_files, embeddings):
    """Parses files using Unstructured and builds the Hybrid RAG indices."""
    docs = []
    temp_dir = tempfile.mkdtemp()
    
    # 1. Ingestion: Parse all unstructured data
    for file in uploaded_files:
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        
        loader = UnstructuredFileLoader(temp_path, strategy="fast")
        docs.extend(loader.load())
        
    # 2. Chunking: 1000 tokens / 200 overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Indexing: Build Vector Store (Chroma)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 4. Indexing: Build Keyword Store (BM25)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 4
    
    # 5. Hybrid Retrieval: Ensemble via Reciprocal Rank Fusion (RRF)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5] # 50% keyword, 50% semantic
    )
    
    return ensemble_retriever

# ==========================================
# 4. STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ==========================================
# 5. SIDEBAR / INGESTION UI
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/1200px-GitHub_Invertocat_Logo.svg.png", width=50) # Placeholder for your brand
    st.title("Data Ingestion")
    
    missing_deps = check_dependencies()
    if missing_deps:
        st.warning("⚠️ **Missing System Dependencies for OCR:**\n\n" + "\n".join(f"- {d}" for d in missing_deps))
        st.caption("Standard text extraction will work, but scanned images may fail.")
        
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, XLSX, TXT)", 
        accept_multiple_files=True,
        type=["pdf", "docx", "xlsx", "txt"]
    )
    
    if st.button("Initialize Hybrid RAG", use_container_width=True, type="primary"):
        if uploaded_files:
            with st.spinner("Processing unstructured data & building indices..."):
                embeddings, _ = load_models()
                st.session_state.retriever = process_documents(uploaded_files, embeddings)
            st.success("✅ Hybrid RAG Ready! Database Indexed.")
        else:
            st.error("Please upload files first.")

# ==========================================
# 6. MAIN UI & CHAT INTERFACE
# ==========================================

# Show Hero Section if no chat history exists
if not st.session_state.messages:
    st.markdown('<p class="hero-title">Meet Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Your AI-Powered Productivity Assistant. 100% Offline & Secure.</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("Answer Questions", "Retrieve instant factual answers backed by your enterprise documentation via semantic search."),
        ("Exact Keyword Lookup", "BM25 integration ensures precise retrieval of IDs, serial numbers, and specific terminology."),
        ("Multi-format Parsing", "Seamlessly read and comprehend PDF, DOCX, XLSX, and TXT files, including image OCR."),
        ("Source Transparency", "Every claim is backed by transparent chunk citations referencing the exact source file.")
    ]
    
    for i, col in enumerate([col1, col2, col3, col4]):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">{features[i][0]}</div>
                <div class="feature-text">{features[i][1]}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br><br>", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Source Citations"):
                for source in msg["sources"]:
                    st.markdown(f'<div class="source-box"><strong>File:</strong> {source["file"]}<br><strong>Content:</strong> {source["content"][:200]}...</div>', unsafe_allow_html=True)

# Handle Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.retriever is None:
        st.error("Please upload and index documents first using the sidebar.")
    else:
        # 1. Display User Message
        st.chat_message("user").markdown(prompt)
        
        # 2. Setup Chain
        _, llm = load_models()
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a highly precise enterprise AI assistant. Answer the user's question based strictly on the provided context. If the answer is not in the context, say 'I cannot find the answer in the provided documents.'\n\nContext:\n{context}"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

        # 3. Stream Output
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Simulated token streaming from the chain
            response = rag_chain.invoke({"input": prompt})
            
            # Extract Sources
            source_docs = response.get("context", [])
            sources_data = []
            for doc in source_docs:
                sources_data.append({
                    "file": doc.metadata.get("source", "Unknown Document"),
                    "content": doc.page_content
                })

            full_response = response["answer"]
            response_placeholder.markdown(full_response)
            
            # Display Sources
            if sources_data:
                with st.expander("View Source Citations"):
                    for src in sources_data:
                        st.markdown(f'<div class="source-box"><strong>File:</strong> {src["file"]}<br><strong>Content:</strong> {src["content"][:300]}...</div>', unsafe_allow_html=True)

        # 4. Save to State
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources_data})