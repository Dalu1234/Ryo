# set_env.py (v4.5 - Revert to all-mpnet-base-v2)
import sys
print("--- EXECUTING set_env.py (v4.5) - Using all-mpnet-base-v2 ---"); sys.stdout.flush()
import os
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader

# --- Initial Debug Prints ---
print(f"--- set_env.py (v4.5) DEBUG: Python executable: {sys.executable}"); sys.stdout.flush()
print(f"--- set_env.py (v4.5) DEBUG: Current working directory: {os.getcwd()}"); sys.stdout.flush()
try:
    print(f"--- set_env.py (v4.5) DEBUG: Files in current directory ({os.getcwd()}): {os.listdir('.')}"); sys.stdout.flush()
    app_allocations_pdf_path = "/app/Allocations.pdf" 
    if os.path.exists(app_allocations_pdf_path):
        print(f"--- set_env.py (v4.5) DEBUG: '{app_allocations_pdf_path}' exists inside the container."); sys.stdout.flush()
    else:
        relative_allocations_pdf_path = "Allocations.pdf"
        if os.path.exists(relative_allocations_pdf_path):
            print(f"--- set_env.py (v4.5) DEBUG: Relative path 'Allocations.pdf' exists in CWD '{os.getcwd()}'."); sys.stdout.flush()
            app_allocations_pdf_path = relative_allocations_pdf_path
        else:
            print(f"--- set_env.py (v4.5) DEBUG: Neither '{app_allocations_pdf_path}' nor relative 'Allocations.pdf' in CWD '{os.getcwd()}' found. Check Dockerfile COPY and PDF location."); sys.stdout.flush()
except Exception as e:
    print(f"--- set_env.py (v4.5) DEBUG: Error listing directory contents: {e}"); sys.stdout.flush()

# === Set General Environment Variables ===
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RyoRAGBot/1.0_Cloud")
print(f"--- set_env.py (v4.5): USER_AGENT set to: {os.environ['USER_AGENT']}"); sys.stdout.flush()

# === Langchain Tracing ===
LANGCHAIN_TRACING_V2_ENV = os.getenv("LANGCHAIN_TRACING_V2", "false") 
if LANGCHAIN_TRACING_V2_ENV.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", f"RyoRAG_AWS_{uuid4().hex[:8]}")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY_CONFIG = os.getenv("LANGCHAIN_API_KEY")
    if LANGCHAIN_API_KEY_CONFIG:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY_CONFIG
        print("--- set_env.py (v4.5): LangSmith tracing configured."); sys.stdout.flush()
    else:
        print("--- set_env.py (v4.5) Warning: LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set in environment."); sys.stdout.flush()
else:
    print("--- set_env.py (v4.5): LangSmith tracing is disabled (LANGCHAIN_TRACING_V2 is not 'true')."); sys.stdout.flush()

# === LLM and Embeddings Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = None
if not GROQ_API_KEY:
    print("--- set_env.py (v4.5) ERROR: GROQ_API_KEY environment variable not set. LLM will not function."); sys.stdout.flush()
else:
    try:
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3, groq_api_key=GROQ_API_KEY)
        print("--- set_env.py (v4.5): Groq LLM initialized."); sys.stdout.flush()
    except Exception as e:
        print(f"--- set_env.py (v4.5) ERROR initializing Groq LLM: {e}"); sys.stdout.flush()

embeddings_model = None
embedding_model_name = "sentence-transformers/all-mpnet-base-v2" # Reverted to original larger model
print(f"--- set_env.py (v4.5): Attempting to load HuggingFace Embeddings model: {embedding_model_name}"); sys.stdout.flush()
try:
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"--- set_env.py (v4.5): HuggingFace Embeddings model '{embedding_model_name}' loaded successfully."); sys.stdout.flush()
except Exception as e:
    print(f"--- set_env.py (v4.5) ERROR initializing HuggingFace Embeddings model '{embedding_model_name}': {e}", exc_info=True); sys.stdout.flush()

# === Qdrant Connection ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
print(f"--- set_env.py (v4.5): Attempting to connect to Qdrant. URL: {QDRANT_URL}, API Key Provided: {'Yes' if QDRANT_API_KEY else 'No'}"); sys.stdout.flush()

client = None
if not QDRANT_URL:
    print("--- set_env.py (v4.5) ERROR: QDRANT_URL environment variable not set. Cannot connect to Qdrant."); sys.stdout.flush()
elif not QDRANT_URL.startswith("https://") and QDRANT_API_KEY :
    print(f"--- set_env.py (v4.5) WARNING: QDRANT_URL '{QDRANT_URL}' is not secure (does not start with https://) but an API key is provided. This is insecure."); sys.stdout.flush()
    client = None
elif not QDRANT_URL.startswith("https://") and not QDRANT_API_KEY: 
    try:
        print(f"--- set_env.py (v4.5): Connecting to Qdrant at {QDRANT_URL} (no API key, assuming local HTTP)."); sys.stdout.flush()
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        client.get_collections() 
        print(f"--- set_env.py (v4.5): Successfully connected to Qdrant at {QDRANT_URL} and listed collections."); sys.stdout.flush()
    except Exception as e:
        print(f"--- set_env.py (v4.5) ERROR connecting to Qdrant at {QDRANT_URL}: {e}"); sys.stdout.flush()
        client = None
else: 
    try:
        if QDRANT_API_KEY:
            print(f"--- set_env.py (v4.5): Connecting to Qdrant Cloud at {QDRANT_URL} with API key."); sys.stdout.flush()
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            print(f"--- set_env.py (v4.5): Connecting to Qdrant Cloud at {QDRANT_URL} (no API key provided, HTTPS)."); sys.stdout.flush()
            client = QdrantClient(url=QDRANT_URL)
        client.get_collections() 
        print(f"--- set_env.py (v4.5): Successfully connected to Qdrant Cloud at {QDRANT_URL} and listed collections."); sys.stdout.flush()
    except Exception as e:
        print(f"--- set_env.py (v4.5) ERROR connecting to Qdrant Cloud at {QDRANT_URL}: {e}"); sys.stdout.flush()
        client = None

collection_name = "allocations_docs" # Reverted to original collection name
uploaded_file_path_in_container = "Allocations.pdf" 
print(f"--- set_env.py (v4.5): PDF path for indexing is set to: {uploaded_file_path_in_container} (expected at /app/{uploaded_file_path_in_container} in container)"); sys.stdout.flush()
print(f"--- set_env.py (v4.5): Using Qdrant collection name: {collection_name}"); sys.stdout.flush()

FORCE_REINDEX_STR = os.getenv("FORCE_REINDEX", "False")
FORCE_REINDEX = FORCE_REINDEX_STR.lower() in ['true', '1', 't']
print(f"--- set_env.py (v4.5): FORCE_REINDEX set to: {FORCE_REINDEX}"); sys.stdout.flush()

vector_store = None
if client and embeddings_model: 
    print("--- set_env.py (v4.5): Qdrant client and embeddings model are available. Proceeding with collection check/setup."); sys.stdout.flush()
    collection_exists_on_startup = False
    try:
        print(f"--- set_env.py (v4.5): Checking if collection '{collection_name}' exists..."); sys.stdout.flush()
        client.get_collection(collection_name=collection_name)
        collection_exists_on_startup = True
        print(f"--- set_env.py (v4.5): Collection '{collection_name}' found on startup."); sys.stdout.flush()
    except Exception as e:
        error_str = str(e).lower()
        if "not found" in error_str or "status_code=404" in error_str or "statuscode.not_found" in error_str :
            print(f"--- set_env.py (v4.5): Collection '{collection_name}' does not exist. Will attempt to create."); sys.stdout.flush()
        else:
            print(f"--- set_env.py (v4.5) Warning: Error trying to get collection '{collection_name}': {e}. Assuming it does not exist for safety."); sys.stdout.flush()
        collection_exists_on_startup = False

    should_index = False
    if FORCE_REINDEX:
        print(f"--- set_env.py (v4.5): FORCE_REINDEX is True."); sys.stdout.flush()
        if collection_exists_on_startup:
            try:
                print(f"--- set_env.py (v4.5): Deleting existing collection '{collection_name}' due to FORCE_REINDEX."); sys.stdout.flush()
                client.delete_collection(collection_name=collection_name)
                print(f"--- set_env.py (v4.5): Collection '{collection_name}' deleted successfully."); sys.stdout.flush()
            except Exception as e:
                print(f"--- set_env.py (v4.5) ERROR deleting collection '{collection_name}': {e}. Proceeding to recreate might fail if delete was incomplete."); sys.stdout.flush()
        should_index = True # Always index if FORCE_REINDEX is true, even if deletion failed
    elif not collection_exists_on_startup:
        print(f"--- set_env.py (v4.5): Collection does not exist, so indexing is required."); sys.stdout.flush()
        should_index = True
    else:
        print(f"--- set_env.py (v4.5): Collection '{collection_name}' exists and FORCE_REINDEX is False. Skipping indexing."); sys.stdout.flush()

    if should_index:
        print(f"--- set_env.py (v4.5): Proceeding with indexing for collection '{collection_name}'..."); sys.stdout.flush()
        actual_pdf_path_in_container = os.path.join(os.getcwd(), uploaded_file_path_in_container) if os.getcwd() == "/app" else uploaded_file_path_in_container
        print(f"--- set_env.py (v4.5): Checking for PDF at resolved path: '{actual_pdf_path_in_container}'"); sys.stdout.flush()
        
        if not os.path.exists(actual_pdf_path_in_container): 
            print(f"--- set_env.py (v4.5) CRITICAL ERROR: PDF file not found at '{actual_pdf_path_in_container}'. Cannot index documents."); sys.stdout.flush()
        else:
            print(f"--- set_env.py (v4.5): PDF file found at '{actual_pdf_path_in_container}'. Proceeding with loading and indexing."); sys.stdout.flush()
            try:
                test_embedding = embeddings_model.embed_query("test query for vector size") 
                actual_vector_size = len(test_embedding)
                print(f"--- set_env.py (v4.5): Determined vector size for '{embedding_model_name}': {actual_vector_size}"); sys.stdout.flush() # Log model name

                print(f"--- set_env.py (v4.5): Recreating collection '{collection_name}' with vector size {actual_vector_size}..."); sys.stdout.flush()
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={"size": actual_vector_size, "distance": "Cosine"}, 
                )
                print(f"--- set_env.py (v4.5): Collection '{collection_name}' created/recreated successfully."); sys.stdout.flush()

                print("--- set_env.py (v4.5): Loading PDF with PyMuPDFLoader..."); sys.stdout.flush()
                loader = PyMuPDFLoader(actual_pdf_path_in_container, extract_images=False)
                docs = loader.load()
                print(f"--- set_env.py (v4.5): Loaded {len(docs)} pages from '{actual_pdf_path_in_container}'."); sys.stdout.flush()

                print("--- set_env.py (v4.5): Splitting documents..."); sys.stdout.flush()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = splitter.split_documents(docs)
                print(f"--- set_env.py (v4.5): Created {len(chunked_docs)} document chunks."); sys.stdout.flush()
                
                if not chunked_docs:
                    print("--- set_env.py (v4.5) Warning: No document chunks were created. Check PDF content and splitter settings."); sys.stdout.flush()
                else:
                    print(f"--- set_env.py (v4.5): Attempting to add {len(chunked_docs)} chunks to Qdrant collection '{collection_name}'..."); sys.stdout.flush()
                    QdrantVectorStore.from_documents(
                        documents=chunked_docs,
                        embedding=embeddings_model, 
                        client=client,
                        collection_name=collection_name,
                    )
                    print(f"--- set_env.py (v4.5): Successfully indexed {len(chunked_docs)} document chunks into '{collection_name}'."); sys.stdout.flush()
                collection_exists_on_startup = True 
            except Exception as e:
                print(f"--- set_env.py (v4.5) CRITICAL ERROR during document loading, processing, or indexing for '{collection_name}': {e}", exc_info=True); sys.stdout.flush()
                collection_exists_on_startup = False 
    
    if collection_exists_on_startup: 
        try:
            print(f"--- set_env.py (v4.5): Initializing query vector_store for collection '{collection_name}'..."); sys.stdout.flush()
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings_model, 
            )
            print("--- set_env.py (v4.5): Query vector_store initialized successfully."); sys.stdout.flush()
        except Exception as e:
            print(f"--- set_env.py (v4.5) ERROR initializing query vector_store for '{collection_name}': {e}", exc_info=True); sys.stdout.flush() 
            vector_store = None
    else:
        print(f"--- set_env.py (v4.5): Query vector_store NOT initialized because collection '{collection_name}' could not be confirmed or created/indexed."); sys.stdout.flush()
        vector_store = None
elif not client:
    print("--- set_env.py (v4.5): Qdrant client not initialized. Vector store will not be available."); sys.stdout.flush()
    vector_store = None
elif not embeddings_model: 
    print("--- set_env.py (v4.5): Embeddings model not initialized. Vector store will not be available."); sys.stdout.flush()
    vector_store = None

if vector_store:
    print("--- set_env.py (v4.5) FINISHED: vector_store IS INITIALIZED ---"); sys.stdout.flush()
else:
    print("--- set_env.py (v4.5) FINISHED: vector_store IS NONE. CHECK LOGS ABOVE FOR ERRORS. ---"); sys.stdout.flush()
