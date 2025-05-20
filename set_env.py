# set_env.py (v4.6 - Fix pickling and print statements)
import sys
from dotenv import load_dotenv # <--- ADD THIS IMPORT
import os # <--- Make sure os is imported before load_dotenv if you specify a path, but usually not an issue for default .env

# Load environment variables from .env file in the current directory (/app)
# This should be one of the first things you do.
load_dotenv() # <--- ADD THIS LINE

print("--- EXECUTING set_env.py (v4.6) - Fixes for pickling and print ---"); sys.stdout.flush()
# import os # Already imported above
import logging
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models # Ensure 'models' is imported for VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader

# --- Initial Debug Prints ---
print(f"--- set_env.py (v4.6) DEBUG: Python executable: {sys.executable}"); sys.stdout.flush()
print(f"--- set_env.py (v4.6) DEBUG: Current working directory: {os.getcwd()}"); sys.stdout.flush()
try:
    print(f"--- set_env.py (v4.6) DEBUG: Files in current directory ({os.getcwd()}): {os.listdir('.')}"); sys.stdout.flush()
    app_allocations_pdf_path = "/app/Allocations.pdf"
    if os.path.exists(app_allocations_pdf_path):
        print(f"--- set_env.py (v4.6) DEBUG: '{app_allocations_pdf_path}' exists inside the container."); sys.stdout.flush()
    else:
        relative_allocations_pdf_path = "Allocations.pdf"
        if os.path.exists(relative_allocations_pdf_path):
            print(f"--- set_env.py (v4.6) DEBUG: Relative path 'Allocations.pdf' exists in CWD '{os.getcwd()}'."); sys.stdout.flush()
            app_allocations_pdf_path = relative_allocations_pdf_path
        else:
            print(f"--- set_env.py (v4.6) DEBUG: Neither '{app_allocations_pdf_path}' nor relative 'Allocations.pdf' in CWD '{os.getcwd()}' found. Check Dockerfile COPY and PDF location."); sys.stdout.flush()
except Exception as e:
    # Use logging for consistency if you prefer, or keep as print for simple debugs
    print(f"--- set_env.py (v4.6) DEBUG: Error listing directory contents: {e}"); sys.stdout.flush()

# === Set General Environment Variables ===
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RyoRAGBot/1.0_Cloud")
print(f"--- set_env.py (v4.6): USER_AGENT set to: {os.environ['USER_AGENT']}"); sys.stdout.flush()

# === Langchain Tracing ===
LANGCHAIN_TRACING_V2_ENV = os.getenv("LANGCHAIN_TRACING_V2", "false")
if LANGCHAIN_TRACING_V2_ENV.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", f"RyoRAG_AWS_{uuid4().hex[:8]}")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY_CONFIG = os.getenv("LANGCHAIN_API_KEY")
    if LANGCHAIN_API_KEY_CONFIG:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY_CONFIG
        print("--- set_env.py (v4.6): LangSmith tracing configured."); sys.stdout.flush()
    else:
        print("--- set_env.py (v4.6) Warning: LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set in environment."); sys.stdout.flush()
else:
    print("--- set_env.py (v4.6): LangSmith tracing is disabled (LANGCHAIN_TRACING_V2 is not 'true')."); sys.stdout.flush()

# === LLM and Embeddings Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = None
if not GROQ_API_KEY:
    print("--- set_env.py (v4.6) ERROR: GROQ_API_KEY environment variable not set. LLM will not function."); sys.stdout.flush()
else:
    try:
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3, groq_api_key=GROQ_API_KEY)
        print("--- set_env.py (v4.6): Groq LLM initialized."); sys.stdout.flush()
    except Exception as e:
        logging.error(f"--- set_env.py (v4.6) ERROR initializing Groq LLM: {e}", exc_info=True); sys.stdout.flush()

embeddings_model = None
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
print(f"--- set_env.py (v4.6): Attempting to load HuggingFace Embeddings model: {embedding_model_name}"); sys.stdout.flush()
try:
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"--- set_env.py (v4.6): HuggingFace Embeddings model '{embedding_model_name}' loaded successfully."); sys.stdout.flush()
except Exception as e:
    logging.error(f"--- set_env.py (v4.6) ERROR initializing HuggingFace Embeddings model '{embedding_model_name}': {e}", exc_info=True); sys.stdout.flush()

# === Qdrant Connection ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PREFER_GRPC = False # Default for HTTPS cloud URLs
if QDRANT_URL and not QDRANT_URL.startswith("http"): # e.g. localhost:6333
    QDRANT_PREFER_GRPC = True
elif QDRANT_URL and QDRANT_URL.startswith("http://") and ":6333" in QDRANT_URL: # e.g. http://localhost:6333
    QDRANT_PREFER_GRPC = True

print(f"--- set_env.py (v4.6): Qdrant params: URL='{QDRANT_URL}', Key Provided={'Yes' if QDRANT_API_KEY else 'No'}, Prefer GRPC={QDRANT_PREFER_GRPC}"); sys.stdout.flush()

client = None # Global client for operations like delete/recreate collection
if QDRANT_URL:
    try:
        print(f"--- set_env.py (v4.6): Initializing global Qdrant client for admin tasks."); sys.stdout.flush()
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY if QDRANT_API_KEY else None, prefer_grpc=QDRANT_PREFER_GRPC)
        client.get_collections() # Test connection
        print(f"--- set_env.py (v4.6): Global Qdrant client connected successfully."); sys.stdout.flush()
    except Exception as e:
        logging.error(f"--- set_env.py (v4.6) ERROR connecting global Qdrant client: {e}", exc_info=True); sys.stdout.flush()
        client = None
else:
    print("--- set_env.py (v4.6) ERROR: QDRANT_URL not set. Qdrant client cannot be initialized."); sys.stdout.flush()


collection_name = "allocations_docs"
uploaded_file_path_in_container = "Allocations.pdf"
print(f"--- set_env.py (v4.6): PDF path for indexing is set to: {uploaded_file_path_in_container} (expected at /app/{uploaded_file_path_in_container} in container)"); sys.stdout.flush()
print(f"--- set_env.py (v4.6): Using Qdrant collection name: {collection_name}"); sys.stdout.flush()

FORCE_REINDEX_STR = os.getenv("FORCE_REINDEX", "False")
FORCE_REINDEX = FORCE_REINDEX_STR.lower() in ['true', '1', 't']
print(f"--- set_env.py (v4.6): FORCE_REINDEX set to: {FORCE_REINDEX}"); sys.stdout.flush()

vector_store = None # This will be initialized for querying later

if client and embeddings_model: # Need client for admin tasks, and embeddings_model for everything
    print("--- set_env.py (v4.6): Qdrant client and embeddings model are available. Proceeding with collection check/setup."); sys.stdout.flush()
    collection_exists_on_startup = False
    try:
        print(f"--- set_env.py (v4.6): Checking if collection '{collection_name}' exists..."); sys.stdout.flush()
        client.get_collection(collection_name=collection_name)
        collection_exists_on_startup = True
        print(f"--- set_env.py (v4.6): Collection '{collection_name}' found on startup."); sys.stdout.flush()
    except Exception as e: # Broad exception for qdrant client errors like collection not found
        error_str = str(e).lower()
        if "not found" in error_str or "status_code=404" in error_str or "statuscode.not_found" in error_str or "collectionnotfoundexception" in error_str:
            print(f"--- set_env.py (v4.6): Collection '{collection_name}' does not exist. Will attempt to create."); sys.stdout.flush()
        else: # Other connection or unexpected error
            logging.warning(f"--- set_env.py (v4.6) Warning: Error checking collection '{collection_name}': {e}. Assuming it does not exist.", exc_info=True); sys.stdout.flush()
        collection_exists_on_startup = False

    should_index = False
    if FORCE_REINDEX:
        print(f"--- set_env.py (v4.6): FORCE_REINDEX is True."); sys.stdout.flush()
        if collection_exists_on_startup:
            try:
                print(f"--- set_env.py (v4.6): Deleting existing collection '{collection_name}' due to FORCE_REINDEX."); sys.stdout.flush()
                client.delete_collection(collection_name=collection_name)
                print(f"--- set_env.py (v4.6): Collection '{collection_name}' deleted successfully."); sys.stdout.flush()
                collection_exists_on_startup = False # It's gone now
            except Exception as e:
                logging.error(f"--- set_env.py (v4.6) ERROR deleting collection '{collection_name}': {e}. Proceeding to recreate might fail.", exc_info=True); sys.stdout.flush()
        should_index = True
    elif not collection_exists_on_startup:
        print(f"--- set_env.py (v4.6): Collection does not exist, so indexing is required."); sys.stdout.flush()
        should_index = True
    else:
        print(f"--- set_env.py (v4.6): Collection '{collection_name}' exists and FORCE_REINDEX is False. Skipping indexing."); sys.stdout.flush()

    # Flag to track if collection is ready for querying (either existed or was successfully indexed)
    collection_ready_for_querying = collection_exists_on_startup and not should_index

    if should_index:
        print(f"--- set_env.py (v4.6): Proceeding with indexing for collection '{collection_name}'..."); sys.stdout.flush()
        actual_pdf_path_in_container = os.path.join(os.getcwd(), uploaded_file_path_in_container) if os.getcwd() == "/app" else uploaded_file_path_in_container
        print(f"--- set_env.py (v4.6): Checking for PDF at resolved path: '{actual_pdf_path_in_container}'"); sys.stdout.flush()

        if not os.path.exists(actual_pdf_path_in_container):
            print(f"--- set_env.py (v4.6) CRITICAL ERROR: PDF file not found at '{actual_pdf_path_in_container}'. Cannot index documents."); sys.stdout.flush()
        else:
            print(f"--- set_env.py (v4.6): PDF file found at '{actual_pdf_path_in_container}'. Proceeding with loading and indexing."); sys.stdout.flush()
            try:
                test_embedding = embeddings_model.embed_query("test query for vector size")
                actual_vector_size = len(test_embedding)
                print(f"--- set_env.py (v4.6): Determined vector size for '{embedding_model_name}': {actual_vector_size}"); sys.stdout.flush()

                # Recreate collection using the global client before indexing
                # This ensures the collection exists with the correct schema
                print(f"--- set_env.py (v4.6): Ensuring collection '{collection_name}' exists with vector size {actual_vector_size}..."); sys.stdout.flush()
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=actual_vector_size, distance=models.Distance.COSINE),
                )
                print(f"--- set_env.py (v4.6): Collection '{collection_name}' created/recreated successfully by global client."); sys.stdout.flush()

                print("--- set_env.py (v4.6): Loading PDF with PyMuPDFLoader..."); sys.stdout.flush()
                loader = PyMuPDFLoader(actual_pdf_path_in_container, extract_images=False)
                docs = loader.load()
                print(f"--- set_env.py (v4.6): Loaded {len(docs)} pages from '{actual_pdf_path_in_container}'."); sys.stdout.flush()

                print("--- set_env.py (v4.6): Splitting documents..."); sys.stdout.flush()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = splitter.split_documents(docs)
                print(f"--- set_env.py (v4.6): Created {len(chunked_docs)} document chunks."); sys.stdout.flush()

                if not chunked_docs:
                    print("--- set_env.py (v4.6) Warning: No document chunks were created. Check PDF content and splitter settings."); sys.stdout.flush()
                else:
                    print(f"--- set_env.py (v4.6): Attempting to add {len(chunked_docs)} chunks to Qdrant collection '{collection_name}' via QdrantVectorStore.from_documents (passing connection params)..."); sys.stdout.flush()
                    # **MODIFIED from_documents call to pass connection parameters directly**
                    QdrantVectorStore.from_documents(
                        documents=chunked_docs,
                        embedding=embeddings_model,
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
                        prefer_grpc=QDRANT_PREFER_GRPC,
                        collection_name=collection_name,
                        # client=client, # Removed: Pass connection params instead
                    )
                    print(f"--- set_env.py (v4.6): Successfully indexed {len(chunked_docs)} document chunks into '{collection_name}'."); sys.stdout.flush()
                collection_ready_for_querying = True # Indexing was attempted and (presumably) succeeded
            except Exception as e:
                logging.error(f"--- set_env.py (v4.6) CRITICAL ERROR during document loading, processing, or indexing for '{collection_name}': {e}", exc_info=True); sys.stdout.flush()
                collection_ready_for_querying = False # Indexing failed

    # Initialize vector_store for querying if collection is ready
    if collection_ready_for_querying:
        try:
            print(f"--- set_env.py (v4.6): Initializing query vector_store for collection '{collection_name}' using from_existing_collection..."); sys.stdout.flush()
            # **USING from_existing_collection as per your finding for the query object**
            vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=collection_name,
                embedding=embeddings_model,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
                prefer_grpc=QDRANT_PREFER_GRPC,
            )
            print(f"--- set_env.py (v4.6): Query vector_store initialized successfully via from_existing_collection for collection '{collection_name}'."); sys.stdout.flush()
        except Exception as e:
            logging.error(f"--- set_env.py (v4.6) ERROR initializing query vector_store via from_existing_collection for '{collection_name}': {e}", exc_info=True); sys.stdout.flush()
            vector_store = None
    else:
        print(f"--- set_env.py (v4.6): Query vector_store NOT initialized because collection '{collection_name}' is not ready (check indexing logs)."); sys.stdout.flush()
        vector_store = None

elif not client: # Global client failed to initialize
    print("--- set_env.py (v4.6): Global Qdrant client not initialized. Vector store will not be available."); sys.stdout.flush()
    vector_store = None
elif not embeddings_model:
    print("--- set_env.py (v4.6): Embeddings model not initialized. Vector store will not be available."); sys.stdout.flush()
    vector_store = None

if vector_store:
    print("--- set_env.py (v4.6) FINISHED: vector_store IS INITIALIZED ---"); sys.stdout.flush()
else:
    print("--- set_env.py (v4.6) FINISHED: vector_store IS NONE. CHECK LOGS ABOVE FOR ERRORS. ---"); sys.stdout.flush()