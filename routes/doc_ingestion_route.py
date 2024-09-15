
import os
from langchain.vectorstores import Chroma
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import os
from flask import request, jsonify, Blueprint, abort
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import tempfile
from utils.env_config import OLLAMA_BASE_URL, PERSIST_DIRECTORY 
from repositories.doc_ingestion_repositories import DocumentRepository

logger = logging.getLogger(__name__)

# Blueprint for routes
doc_ingestion_route_blueprint = Blueprint('doc_ingestion_route_blueprint', __name__)
is_pdf_stored_locally = False


# Custom PDF loader for PDFs
class PDFLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load(self):
        docs = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.folder_path, filename)
                docs.extend(self.load_pdf(file_path))
        return docs

    def load_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return [Document(page_content=text, metadata={"source": file_path})]
        
    def load_pdfs_from_request(self, files):
        documents = []
        
        for file in files:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file.seek(0)
                
                # Load PDF content
                reader = PdfReader(temp_file.name)
                pdf_content = ""
                for page in reader.pages:
                    pdf_content += page.extract_text()
                    
                logger.debug(pdf_content,"--")
                documents.extend([Document(page_content = pdf_content, metadata =  {"source": file.filename})])
                
        return documents
            
@doc_ingestion_route_blueprint.route('/ingest-files', methods=['POST'])
def ingest_documents():
    
    index_name = request.form.get('index_name', 'default_index')
    uploaded_files = request.files.getlist('files')
    
    if len(uploaded_files)==0:
        abort(400, "No files provided to ingest")
    
    if is_pdf_stored_locally:
        pdf_folder_path = "/Users/shivamchoudhary/Documents/gpt-voice-assistant/gpt-voice-assistant/pdfs"
        
        # Load PDFs
        pdf_loader = PDFLoader(pdf_folder_path)
        pdf_documents = pdf_loader.load()
    else:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        pdf_loader = PDFLoader(folder_path="")
        pdf_documents = pdf_loader.load_pdfs_from_request(uploaded_files)

    documents = pdf_documents
    logger.debug("Pdf data is extracted into txt format")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = text_splitter.split_documents(documents)

    # Initialize HuggingFaceEmbeddings for the SentenceTransformer model
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Define the persistence directory in a writable path
    persist_directory = PERSIST_DIRECTORY

    # Create embeddings using the HuggingFaceEmbeddings model and store in ChromaDB
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name = index_name
    )

    # Persist the vector database
    vectordb.persist()
    vectordb = None 
    
    collection_repository = DocumentRepository()
    collection_repository.create_document(
        storage_index=index_name, 
        name=f"{index_name}_collection", 
        ingested_files=[{"fileName": file.filename, "isSuccess": True} for file in uploaded_files]
    )

    return jsonify({"message": "Documents ingested successfully!"}), 200


@doc_ingestion_route_blueprint.route('/query', methods=['POST'])
def query_documents():
    
    index_name = request.form.get('index_name', 'default_index')
    query = request.form.get('q')
    
    if not query:
        abort(400, "Query is not provided")
        
    logger.debug(f"Query: {query} is asked on index: {index_name}")
    
    if not OLLAMA_BASE_URL or not PERSIST_DIRECTORY:
        raise EnvironmentError("Environment variables 'OLLAMA_BASE_URL' or 'PERSIST_DIRECTORY' are not set.")

    # Define the persistence directory in a writable path
    persist_directory = PERSIST_DIRECTORY

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedder,
        collection_name=index_name
    )

    retriever = vectordb.as_retriever()
    results = retriever.invoke(input=query)
    
    context = "\n\n".join([result.page_content for result in results])

    # Build the prompt for Ollama
    prompt_template = f"""
    You are an AI assistant. Here is some relevant context extracted from documents:
    
    {context}
    
    Now, based on this context, please provide an answer to the following query:
    
    Query: {query}
    """

    # Prepare payload for Ollama API
    ollama_payload = {
        "model": "llama3.1",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that uses relevant context to answer questions. \n"
                "If Answering can't be possible with given context, simply say - Answer not in context. \n"
                "Always use provided context for response generation."
            },
            {
                "role": "user",
                "content": prompt_template
            }
        ],
        "stream": False
    }

    # Send the request to the Ollama API
    ollama_response = requests.post(url=OLLAMA_BASE_URL, json=ollama_payload)

    # Check if the request was successful
    if ollama_response.status_code == 200:
        response_data = ollama_response.json()
        return jsonify({
            "query": query,
            "response": response_data['message']['content']
        }), 200
    else:
        return jsonify({"error": "Failed to get response from Ollama."}), ollama_response.status_code
    
@doc_ingestion_route_blueprint.route('/get-indexes', methods=['GET'])
def get_indexes():
    try:
        # Initialize the repository
        collection_repository = DocumentRepository()
        
        # Retrieve collection names
        documents = collection_repository.get_collection_names_with_storage_index()
        
        # Return the collections as a JSON response
        return documents, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
