
import os
from flask import request, jsonify, Blueprint
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import os
from flask import request, jsonify, Blueprint
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.docstore.document import Document

# Blueprint for routes
doc_ingestion_route_blueprint = Blueprint('doc_ingestion_route_blueprint', __name__)

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

# Route to ingest files
@doc_ingestion_route_blueprint.route('/ingest-files', methods=['POST'])
def ingest_documents():
    pdf_folder_path = "/Users/shivamchoudhary/Documents/gpt-voice-assistant/gpt-voice-assistant/pdfs"
    
    # Load PDFs and text documents
    pdf_loader = PDFLoader(pdf_folder_path)
    pdf_documents = pdf_loader.load()

    # # Load text documents
    # loader = DirectoryLoader(pdf_folder_path, glob="*.txt", loader_cls=TextLoader)
    # text_documents = loader.load()

    # Combine both PDF and text documents
    documents = pdf_documents

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = text_splitter.split_documents(documents)

    # Initialize SentenceTransformer model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Define the persistence directory for Chroma
    persist_directory = '/content/chroma_db_persistence'

    # Create embeddings using SentenceTransformer and store in ChromaDB
    embeddings = [embedder.encode(text.page_content) for text in texts]

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Persist data
    vectordb.persist()
    vectordb = None  # Closing the connection
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Reload the vector database
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=lambda texts: [embedder.encode(text) for text in texts]
    )

    # Convert the query into an embedding using SentenceTransformer
    query="family and medical leave policy"
    query_embedding = embedder.encode(query)

    # Query the vector database with the generated embedding
    retriever = vectordb.as_retriever()
    results = retriever.get_relevant_documents(query_embedding, k=5)

    return jsonify({
        "query": query,
        "results": [
            {"source": result.metadata['source'], "content": result.page_content[:500]}
            for result in results
        ]
    }), 200



@doc_ingestion_route_blueprint.route('/query', methods=['POST'])
def query_documents():
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "Query text is required"}), 400

    # Reload the vector store
    persist_directory = '/content/chroma_db_persistence'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding=embedding)

    # Query the vector database
    retriever = vectordb.as_retriever()
    results = retriever.get_relevant_documents(query, k=5)

    # Return the results as JSON
    return jsonify({
        "query": query,
        "results": [
            {"source": result.metadata['source'], "content": result.page_content[:500]}
            for result in results
        ]
    }), 200
