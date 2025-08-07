import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_pdf_files():
    """Get list of PDF files in the research_papers directory"""
    if not os.path.exists("research_papers"):
        os.makedirs("research_papers")
        return []
    
    pdf_files = [f for f in os.listdir("research_papers") if f.lower().endswith('.pdf')]
    return sorted(pdf_files)

def load_selected_pdfs(selected_files):
    """Load selected PDF files"""
    all_docs = []
    
    for filename in selected_files:
        try:
            file_path = os.path.join("research_papers", filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add source filename to each document
            for doc in docs:
                doc.metadata['source'] = filename
            
            all_docs.extend(docs)
            st.write(f" Loaded: {filename}")
            
        except Exception as e:
            st.write(f" Error loading {filename}: {str(e)}")
    
    return all_docs

# Get API keys
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Initialize the LLM
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

def create_vector_embedding(selected_files):
    """Create vector embeddings from selected PDF documents"""
    try:
        if not selected_files:
            st.error("Please select at least one PDF file!")
            return False
        
        with st.spinner(f"Processing {len(selected_files)} selected files..."):
            # Initialize embeddings
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Load selected documents
            docs = load_selected_pdfs(selected_files)
            
            if not docs:
                st.error("No documents could be loaded!")
                return False
            
            st.success(f"Loaded {len(docs)} pages from {len(selected_files)} files")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs)
            
            st.info(f"Created {len(final_documents)} text chunks")
            
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(
                final_documents,
                st.session_state.embeddings
            )
            
            # Save selected files info
            st.session_state.selected_files = selected_files
            
        return True
        
    except Exception as e:
        st.error(f"Error creating vector embeddings: {str(e)}")
        return False

def main():
    """Main Streamlit application"""
    st.title("RAG Document Q&A with Groq and Llama3")
    st.markdown("Select PDF files and ask questions about them!")
    
    # Sidebar
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Place PDF files in 'research_papers' folder
        2. Select which PDFs to use
        3. Create embeddings
        4. Ask questions!
        """)
        
        if "vectors" in st.session_state:
            st.success(" Ready to answer questions!")
            if "selected_files" in st.session_state:
                st.write("**Processing:**")
                for file in st.session_state.selected_files:
                    st.write(f"• {file}")
        else:
            st.warning(" Select and process PDFs first")
    
    # Get available PDF files
    pdf_files = get_pdf_files()
    
    if not pdf_files:
        st.warning("No PDF files found in 'research_papers' folder!")
        st.info("Add PDF files to the 'research_papers' folder and refresh the page.")
        return
    
    # PDF Selection Section
    st.subheader(" Select PDF Files")
    st.write(f"Found {len(pdf_files)} PDF files:")
    
    # Simple multiselect for PDF files
    selected_files = st.multiselect(
        "Choose PDFs to process:",
        options=pdf_files,
        default=pdf_files,  # Select all by default
        help="Select one or more PDF files to analyze"
    )
    
    # Show selected files count
    if selected_files:
        st.success(f"Selected {len(selected_files)} out of {len(pdf_files)} files")
    else:
        st.warning("No files selected!")
    
    # Process button
    st.subheader(" Process Documents")
    
    if st.button(" Create Embeddings", type="primary"):
        if selected_files:
            success = create_vector_embedding(selected_files)
            if success:
                st.balloons()
                st.success("Embeddings created successfully!")
        else:
            st.error("Please select at least one PDF file!")
    
    # Query section
    st.subheader(" Ask Questions")
    
    user_prompt = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic discussed in the documents?"
    )
    
    if user_prompt:
        if "vectors" not in st.session_state:
            st.warning("Please create embeddings first!")
        else:
            try:
                with st.spinner("Finding answer..."):
                    # Create chains
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    # Get response
                    start_time = time.process_time()
                    response = retrieval_chain.invoke({'input': user_prompt})
                    response_time = time.process_time() - start_time
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(response['answer'])
                    
                    st.info(f"Response time: {response_time:.2f} seconds")
                    
                    # Show sources
                    with st.expander(" Source Documents"):
                        for i, doc in enumerate(response['context']):
                            source = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**Source {i+1}: {source}**")
                            st.text(doc.page_content)
                            st.markdown("---")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()