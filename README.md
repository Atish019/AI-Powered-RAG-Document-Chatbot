#  RAG Document Q&A with Groq and Llama3

A powerful Retrieval-Augmented Generation (RAG) application that allows you to upload, select, and intelligently query your PDF documents using Groq's Llama3 model and advanced vector search capabilities.

##  Demo Screenshot

Here is the interface of the application:

![App Screenshot](./images\Screenshot%202025-08-07%20003750.png)


##  Project Workflow Diagram

The following diagram shows how the **AI-Powered RAG Document Chatbot** works:

![Project Workflow](./images\Workflow%20daigram.png)


##  Key Features

###  **Flexible Document Management**
- **Direct Upload**: Upload PDF files through the web interface
- **Manual Upload**: Place files in the `research_papers/` directory
- **Smart Detection**: Automatic PDF file discovery and validation

###  **Advanced PDF Selection**
- **Select All**: Process all available PDF documents
- **Select Specific Files**: Choose individual documents with multiselect
- **Select by Pattern**: Filter files using keywords 

###  **Intelligent Q&A System**
- **Context-Aware Answers**: Responses based only on your selected documents
- **Source Tracking**: See exactly which document(s) each answer comes from
- **Fast Retrieval**: FAISS-powered vector search for quick results

###  **User-Friendly Interface**
- **Real-time Status**: Track processing progress and system status
- **Progress Indicators**: Visual feedback during document processing
- **Source Citations**: Expandable view of source documents used in answers
- **Performance Metrics**: Response time tracking

##  Project Structure

AI-Powered-RAG-Document-Chatbot
│
├── app.py                   
├── requirements.txt          
├── .env                     
├── .gitignore              
├── README.md              
├── setup.py               
│
├── research_papers/        

##  Quick Start

### 1. Installation

#### Option A: Automated Setup

# Clone or download the project
git clone <repository-url>


# Run the automated setup
python setup.py

# Install dependencies
pip install -r requirements.txt

### 2. Configuration

#### Set Up Environment Variables
Create a `.env` file in the project root:

# Required: Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: HuggingFace tokens (for advanced features)
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
HF_TOKEN=your_hf_token_here


#### Get Your Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Create an account (free)
3. Generate an API key
4. Add it to your `.env` file

### 3. Run the Application

streamlit run app.py

The application will open in your browser at `http://localhost:8501`

##  How to Use

### Step 1: Add PDF Documents

#### Method A: Direct Upload (Recommended)
1. Use the " Upload PDF Documents" section in the app
2. Select one or more PDF files
3. Files are automatically saved to `research_papers/`

#### Method B: Manual Upload
1. Place PDF files in the `research_papers/` directory
2. Refresh the app to detect new files

### Step 2: Select Documents

Choose your selection method:

- ** Select All**: Process all available PDF files
- ** Select Specific Files**: Choose individual documents
- ** Select by Pattern**: Filter by keyword 

### Step 3: Create Embeddings

1. Click " Create Embeddings for Selected Files"
2. Wait for processing to complete
3. System will show progress for each file

### Step 4: Ask Questions

1. Enter your question in the text input
2. Get AI-powered answers based on your selected documents
3. View source documents to verify information

##  Configuration Options

### Document Processing Settings
```python
# In create_vector_embedding() function
chunk_size=1000          
chunk_overlap=200        
max_pages=200           
```

### Embedding Model Settings
```python
# Default embedding model
model_name="all-MiniLM-L6-v2"  

# Alternative models (larger, more accurate)
# model_name="all-mpnet-base-v2"
# model_name="sentence-transformers/all-MiniLM-L12-v2"
```


##  Supported File Types

###  Fully Supported
- **PDF files** (.pdf extension)
- Text-based PDFs with readable content
- Multi-page documents
- Academic papers, reports, articles

###  Not Supported
- Scanned PDFs (image-only)
- Password-protected PDFs
- Corrupted PDF files
- Non-PDF formats (Word, PowerPoint, etc.)


#### For Better Accuracy
- Use larger embedding models
- Increase chunk overlap
- Reduce chunk size for more granular search

##  Technical Details

### Architecture Overview
```
User Query → Document Selection → Vector Search → LLM Processing → Response
     ↓              ↓                    ↓              ↓            ↓
   Input       File Filtering      FAISS Index     Groq API    Final Answer
```

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **LLM**: Groq Llama3-8B-8192 (fast inference)
- **Embeddings**: HuggingFace sentence-transformers
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF (PDF text extraction)
- **Framework**: LangChain (RAG orchestration)

### Data Flow
1. **Document Upload**: PDFs saved to local directory
2. **Selection**: User chooses specific files to process
3. **Text Extraction**: PyPDF extracts text from selected PDFs
4. **Chunking**: Text split into overlapping chunks
5. **Embedding**: Chunks converted to vector representations
6. **Storage**: Vectors stored in FAISS index
7. **Query Processing**: User questions embedded and matched
8. **Response**: Relevant chunks sent to LLM for answer generation

##  Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --server.runOnSave true
```

### Code Structure
- **`app.py`**: Main application logic and UI
- **`requirements.txt`**: Python dependencies
- **`setup.py`**: Automated project setup
- **`.env`**: Environment configuration
- **`utils/`**: Utility functions (optional)


### Resources
- [Groq Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)

##  Acknowledgments

- **[Groq](https://groq.com/)** - Fast LLM inference platform
- **[LangChain](https://langchain.com/)** - RAG framework and orchestration
- **[HuggingFace](https://huggingface.co/)** - Embedding models and transformers
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[FAISS](https://faiss.ai/)** - Efficient vector similarity search
- **[PyPDF](https://pypdf.readthedocs.io/)** - PDF text extraction
