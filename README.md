# Offline Document Assistant

Privacy-First AI Document Question Answering Using Hybrid Retrieval

## Overview

Offline Document Assistant is a privacy-focused AI application that allows users to interact with their documents using natural language.

The application uses Retrieval-Augmented Generation to retrieve relevant information from uploaded documents and generate context-aware answers using a locally running Large Language Model.

All document processing, retrieval, embedding generation, and language model inference are performed locally. Documents do not need to be uploaded to external cloud AI services.

The application supports PDF, DOCX, XLSX, and TXT files.

## Why This Project

Documents often contain large amounts of information that can be difficult and time-consuming to search manually.

Cloud-based AI document assistants can simplify this process, but they may introduce privacy concerns, API costs, and dependency on external services.

This project addresses these problems by providing a local document question answering system.

The main objectives are:

* Keep documents on the user's local machine
* Avoid external AI API dependencies
* Provide natural language document search
* Combine semantic and keyword-based retrieval
* Generate answers using a local Large Language Model
* Provide retrieved document context for answer verification

## Key Features

### Privacy-First Architecture

The application is designed to run locally.

Documents remain on the user's machine and are processed locally.

No external AI API key is required for the core RAG pipeline.

### Multi-Format Document Support

The application supports:

* PDF
* DOCX
* XLSX
* TXT

Document parsing is handled using the Unstructured library.

### Hybrid Retrieval

The system combines two retrieval approaches:

* Semantic vector search using ChromaDB
* Keyword search using BM25

The combination improves retrieval for both conceptually similar queries and exact keyword searches.

### Local Large Language Model

The application uses Ollama to run the Llama 3.2 model locally.

This allows the document question answering pipeline to operate without sending document content to a cloud-based LLM.

### Local Embeddings

The application uses the nomic-embed-text model through Ollama to generate document embeddings.

### Streaming Responses

AI responses are displayed progressively in the Streamlit interface to provide a more interactive experience.

### Retrieved Context Viewer

The application displays the retrieved document chunks used to generate an answer.

This makes it easier to understand which parts of the document were used by the retrieval system.

### Conversation History

The Streamlit session maintains the conversation history while the application is running.

## Technology Stack

| Component            | Technology                        | Purpose                                |
| -------------------- | --------------------------------- | -------------------------------------- |
| Frontend             | Streamlit                         | Web interface                          |
| Programming Language | Python                            | Application development                |
| LLM                  | Ollama with Llama 3.2             | Local text generation                  |
| Embeddings           | nomic-embed-text                  | Document embeddings                    |
| Framework            | LangChain                         | RAG pipeline and component integration |
| Vector Database      | ChromaDB                          | Semantic vector search                 |
| Keyword Retriever    | BM25                              | Exact keyword retrieval                |
| Document Parser      | Unstructured                      | Document extraction                    |
| Text Splitter        | Recursive Character Text Splitter | Document chunking                      |

## System Architecture

The application follows a Retrieval-Augmented Generation architecture.

```text
User
 |
 v
Streamlit Interface
 |
 v
Document Upload
 |
 v
Unstructured Document Parser
 |
 v
Text Splitting
 |
 +-------------------------+
 |                         |
 v                         v
ChromaDB Vector Search    BM25 Keyword Search
 |                         |
 +------------+------------+
              |
              v
       Hybrid Retrieval
              |
              v
      Relevant Document Chunks
              |
              v
        Prompt Construction
              |
              v
        Ollama Llama 3.2
              |
              v
        Generated Answer
              |
              v
       Streamlit Interface
```

## How the Application Works

### Step 1: Document Upload

The user uploads a supported document through the Streamlit interface.

Supported formats include PDF, DOCX, XLSX, and TXT.

### Step 2: Document Parsing

The uploaded file is temporarily stored and processed using the Unstructured document loader.

The loader extracts the textual content from the document.

### Step 3: Text Chunking

The extracted content is divided into smaller overlapping chunks using the Recursive Character Text Splitter.

Chunking allows the retrieval system to work with manageable pieces of information instead of processing an entire document at once.

The current implementation uses:

```text
Chunk size: 1000 characters
Chunk overlap: 200 characters
```

### Step 4: Embedding Generation

The document chunks are converted into numerical vector representations using the local nomic-embed-text model.

These vectors represent the semantic meaning of the document content.

### Step 5: Vector Storage

The generated embeddings are stored in ChromaDB.

ChromaDB provides semantic similarity search over the document chunks.

### Step 6: Keyword Indexing

The same document chunks are indexed using the BM25 retrieval algorithm.

BM25 is useful when the user's query contains exact words, technical terms, identifiers, names, or phrases that should be matched directly.

### Step 7: Hybrid Retrieval

When a user asks a question, both retrievers are used:

```text
User Question
 |
 +----------------------+
 |                      |
 v                      v
BM25 Search       Chroma Semantic Search
 |                      |
 +----------+-----------+
            |
            v
      Combined Results
            |
            v
     Relevant Chunks
```

The system combines the results from both retrieval methods using an ensemble retriever.

### Step 8: Context Construction

The retrieved document chunks are combined into a context that is provided to the language model.

### Step 9: Answer Generation

The local Llama 3.2 model receives the question and retrieved context.

The model generates the final answer based on the available document information.

### Step 10: Source Display

The application displays the retrieved document chunks so the user can inspect the information that was provided to the language model.

## Why Hybrid Retrieval

A single retrieval technique is not always sufficient for document question answering.

Vector search is useful for understanding semantic similarity.

For example, a user may ask:

```text
What are the responsibilities of the maintenance department?
```

while the document contains:

```text
Duties and functions of the maintenance division
```

Semantic search can identify the relationship between these concepts.

However, vector search may not always perform well for exact terms such as:

```text
BHEL
ISO 9001
Purchase Order 12345
Section 4.2.1
```

BM25 is particularly useful for these types of exact keyword searches.

The project therefore combines both approaches.

```text
Semantic Search
+
Keyword Search
=
Hybrid Retrieval
```

This provides a more robust retrieval strategy for technical and enterprise documents.

## Project Structure

```text
Offline-Document-Assistant/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- assets/
|
|-- data/
|
|-- chroma_db/
|
|-- venv/
```

### File Description

`app.py`

Main Streamlit application containing the user interface, document processing pipeline, retriever configuration, and response generation.

`requirements.txt`

Contains the Python dependencies required to run the project.

`README.md`

Project documentation.

`.gitignore`

Prevents unnecessary files such as the virtual environment, cache files, and local databases from being committed to Git.

`data/`

Directory for local document files.

`chroma_db/`

Directory reserved for ChromaDB persistent storage when persistent vector storage is enabled.

`assets/`

Contains optional application assets such as logos and images.

## Installation

### Prerequisites

The following software is recommended:

* Python 3.11 or newer
* Ollama
* At least 8 GB RAM
* Sufficient disk space for local AI models

### Clone the Repository

```bash
git clone https://github.com/yourusername/Offline-Document-Assistant.git
cd Offline-Document-Assistant
```

Replace `yourusername` with your GitHub username.

### Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama on your computer.

After installation, download the required models:

```bash
ollama pull llama3.2
```

```bash
ollama pull nomic-embed-text
```

Verify the installed models:

```bash
ollama list
```

Make sure both models are available before running the application.

## Running the Application

Activate the virtual environment first.

Windows:

```bash
venv\Scripts\activate
```

Then start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Using the Application

### Upload a Document

Open the application and upload a PDF, DOCX, XLSX, or TXT file from the sidebar.

### Wait for Indexing

The application will:

1. Parse the document
2. Split the content into chunks
3. Generate embeddings
4. Create the vector search index
5. Create the BM25 keyword index
6. Configure the hybrid retriever

### Ask Questions

After indexing is complete, enter a question in the chat interface.

For example:

```text
What is the main purpose of this document?
```

Or:

```text
What are the responsibilities mentioned in section 3?
```

Or:

```text
Summarize the important points of the document.
```

### Inspect Retrieved Context

After receiving an answer, open the retrieved context section to inspect the document chunks used during retrieval.

## Current Configuration

The current implementation uses the following retrieval configuration:

```text
Vector Retriever:
Top K = 3

BM25 Retriever:
Top K = 3

Hybrid Retriever:
BM25 Weight = 0.5
Vector Search Weight = 0.5
```

Text splitting configuration:

```text
Chunk Size = 1000
Chunk Overlap = 200
```

Local models:

```text
LLM = llama3.2
Embedding Model = nomic-embed-text
```

## Privacy

Privacy is one of the primary design goals of this project.

The core pipeline is designed around local processing.

Documents are processed on the user's computer.

The language model runs through Ollama locally.

Embeddings are generated locally using Ollama.

The retrieval database runs locally.

No external AI API is required for the core application.

Users should still review the configuration of all third-party libraries and their own network environment when deploying the system in sensitive environments.

## Limitations

The current version is an early implementation and has several limitations.

The vector database is not yet configured as a fully persistent enterprise knowledge base.

The application currently focuses on a single uploaded document during a session.

PDF page-level citations are not yet implemented.

Scanned PDFs requiring OCR may require additional processing.

Very large documents may require optimized indexing and chunking strategies.

The quality of generated answers depends on the quality of document extraction, retrieval, and the local language model.

## Future Roadmap

### Document Processing

* Multi-document support
* Improved PDF extraction
* OCR for scanned documents
* Improved table extraction
* Image and diagram processing

### Retrieval

* Persistent ChromaDB
* Query rewriting
* Hybrid retrieval optimization
* Cross-encoder reranking
* Parent document retrieval
* Metadata filtering

### Generation

* Improved prompt engineering
* Structured responses
* Conversation-aware retrieval
* Local conversation memory
* Hallucination reduction

### Citations

* Page-level PDF citations
* Document name references
* Chunk-level citations
* Source highlighting
* Citation verification

### Application

* Document management
* Chat export
* PDF report generation
* Authentication
* User-specific document collections
* Enterprise knowledge bases

### Deployment

* Docker support
* Local server deployment
* Enterprise deployment configuration
* Monitoring and logging
* Performance optimization

## Version Roadmap

### Version 1.0

Basic offline document question answering.

Implemented features include:

* Local LLM
* Document upload
* Document parsing
* Text chunking
* Vector retrieval
* BM25 retrieval
* Hybrid retrieval
* Streamlit interface

### Version 2.0

Planned modernization of the architecture.

Planned improvements include:

* Modular RAG architecture
* Multi-document support
* Persistent vector database
* Better document processing
* Improved retrieval
* Citations
* Reranking
* Query rewriting
* Better source tracking
* Improved user interface

### Future Enterprise Version

The long-term goal is to evolve the project into an offline enterprise document intelligence platform.

The enterprise version will focus on:

* Secure document processing
* Local LLM infrastructure
* Advanced retrieval
* Document-level and page-level citations
* Multiple document collections
* Access control
* Audit logging
* Enterprise knowledge bases
* Scalable local deployment

## Contributing

Contributions are welcome.

To contribute:

```bash
git checkout -b feature/your-feature
```

Make your changes and test them locally.

Commit your changes:

```bash
git add .
git commit -m "Add your feature"
```

Push your branch:

```bash
git push origin feature/your-feature
```

Then create a Pull Request on GitHub.

## Development Guidelines

Follow Python best practices.

Keep functions focused on a single responsibility.

Use meaningful variable and function names.

Add comments where the implementation is not immediately obvious.

Update the documentation when adding major features.

Test changes before creating a Pull Request.

Avoid committing virtual environments, local databases, private documents, or secrets.

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute the project according to the terms of the MIT License.

## Author

Piyush Upadhyay

Information Technology Undergraduate

Areas of Interest:

* Artificial Intelligence
* Retrieval-Augmented Generation
* Large Language Models
* Natural Language Processing
* Full Stack Development

## Acknowledgements

This project uses several open-source technologies:

* Python
* Streamlit
* LangChain
* Ollama
* ChromaDB
* Unstructured
* BM25
* nomic-embed-text
* Llama 3.2

## Project Status

Current Status: Working Prototype

The current version demonstrates an offline document question answering pipeline using local LLM inference and hybrid retrieval.

The project is actively being improved toward a modular Offline Enterprise RAG System.

## Summary

Offline Document Assistant demonstrates how Retrieval-Augmented Generation can be implemented locally without depending on cloud-based AI APIs.

The project combines document parsing, text chunking, local embeddings, vector search, keyword retrieval, hybrid retrieval, and local LLM generation into a single application.

The next major development stage is Version 2.0, which will transform the current prototype into a more modular, scalable, and enterprise-oriented offline RAG architecture.
