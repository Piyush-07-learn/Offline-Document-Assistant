import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains.question_answering import load_qa_chain

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise PDF Assistant",
    page_icon="💼",
    layout="wide"
)

# 2. Custom CSS Injection for Modern UI Styling
st.markdown("""
<style>
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make the chat input box float nicely with a modern shadow */
    .stChatInput {
        border-radius: 20px !important;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Subtle border for sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #333;
    }
    
    /* Styled file drag-and-drop box */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #636EFA;
        border-radius: 15px;
        background-color: rgba(99, 110, 250, 0.05);
    }
    
    /* Expandable source dropdown styling */
    .streamlit-expanderHeader {
        background-color: #1E1E24;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Enterprise Local PDF Assistant")
st.caption("Secure, offline AI intelligence with interactive follow-up suggestions.")

# 3. Sidebar Control Panel
with st.sidebar:
    st.header("⚙️ Document & Controls")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    st.divider()
    
    # Button to clear chat history
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. Initialize Interactive Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display source citations if attached to the message
        if "sources" in message and message["sources"]:
            with st.expander("📍 View Verification Sources & Page Numbers"):
                for src in message["sources"]:
                    st.markdown(f"**Page {src['page']}:**\n_{src['content']}_")

# 6. Core Processing & Query Handling
if uploaded_file:
    
    # Cached PDF Indexing Function with Safety Checks
    @st.cache_resource(show_spinner="Indexing document into vector database...")
    def process_pdf(file_bytes, filename):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file_bytes)
            file_path = tf.name

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        os.remove(file_path)

        # Safety Check: Did we actually extract any readable text?
        if not chunks:
            return None, 0

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(chunks, embeddings)
        
        return vector_store, len(docs)

    # Process and retrieve vector database
    vector_store, total_pages = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
    
    # Handle Scanned/Empty PDFs Gracefully
    if vector_store is None:
        st.sidebar.error(f"Error: Could not extract text from **{uploaded_file.name}**. Please ensure it is a text-based PDF, not an image scan.")
        st.stop()
        
    st.sidebar.success(f"Ready: **{uploaded_file.name}** ({total_pages} pages)")

    # Interactive Chat Input
    if query := st.chat_input("Ask a question about your document..."):
        
        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Assistant Answer with Follow-Up Prompting
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document and generating response..."):
                llm = OllamaLLM(model="llama3.2")
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Retrieve top 3 relevant chunks from ChromaDB
                matching_docs = vector_store.similarity_search(query, k=3)
                
                # Instruction to force the AI to ask follow-up questions
                interactive_query = (
                    f"{query}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Provide a direct and complete answer based on the context.\n"
                    "2. At the end of your answer, add a new section starting with '**💡 Suggested Follow-up Questions:**'. "
                    "In this section, suggest 1 to 2 relevant follow-up questions the user could ask next to explore this topic further."
                )
                
                response = chain.invoke({"input_documents": matching_docs, "question": interactive_query})
                answer = response["output_text"]
                
                st.markdown(answer)
                
                # Extract page numbers and content snippets for auditability
                sources = [
                    {
                        "page": doc.metadata.get("page", 0) + 1,
                        "content": doc.page_content[:250].replace("\n", " ") + "..."
                    }
                    for doc in matching_docs
                ]
                
                # Display Expandable Sources Dropdown
                if sources:
                    with st.expander("📍 View Verification Sources & Page Numbers"):
                        for src in sources:
                            st.markdown(f"**Page {src['page']}:**\n_{src['content']}_")

        # Save assistant response and sources to session state history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources
        })

else:
    st.info("👈 Please upload a PDF document in the sidebar to begin analysis.")