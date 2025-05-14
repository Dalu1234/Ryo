# set_env.py (v4)
print("--- EXECUTING set_env.py (v4) - THIS IS THE CORRECT VERSION ---") # New prominent first line
import os
import sys # For path debugging
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader

# --- Initial Debug Prints ---
print(f"--- set_env.py (v4) DEBUG: Python executable: {sys.executable}")
print(f"--- set_env.py (v4) DEBUG: Current working directory: {os.getcwd()}")
try:
    print(f"--- set_env.py (v4) DEBUG: Files in current directory ({os.getcwd()}): {os.listdir('.')}")
    # Check for Allocations.pdf specifically in /app
    app_allocations_pdf_path = "/app/Allocations.pdf"
    if os.path.exists(app_allocations_pdf_path):
        print(f"--- set_env.py (v4) DEBUG: '{app_allocations_pdf_path}' exists inside the container.")
    else:
        print(f"--- set_env.py (v4) DEBUG: '{app_allocations_pdf_path}' DOES NOT exist. Check Dockerfile COPY and PDF location.")
except Exception as e:
    print(f"--- set_env.py (v4) DEBUG: Error listing directory contents: {e}")

# === Set General Environment Variables ===
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RyoRAGBot/1.0_Cloud")
print(f"--- set_env.py (v4): USER_AGENT set to: {os.environ['USER_AGENT']}")

# === Langchain Tracing (Optional but good practice) ===
LANGCHAIN_TRACING_V2_ENV = os.getenv("LANGCHAIN_TRACING_V2", "false") # Default to false if not set
if LANGCHAIN_TRACING_V2_ENV.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", f"RyoRAG_AWS_{uuid4().hex[:8]}")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY_CONFIG = os.getenv("LANGCHAIN_API_KEY")
    if LANGCHAIN_API_KEY_CONFIG:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY_CONFIG
        print("--- set_env.py (v4): LangSmith tracing configured.")
    else:
        print("--- set_env.py (v4) Warning: LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set in environment.")
else:
    print("--- set_env.py (v4): LangSmith tracing is disabled (LANGCHAIN_TRACING_V2 is not 'true').")


# === LLM and Embeddings Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = None
if not GROQ_API_KEY:
    print("--- set_env.py (v4) ERROR: GROQ_API_KEY environment variable not set. LLM will not function.")
else:
    try:
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3, groq_api_key=GROQ_API_KEY)
        print("--- set_env.py (v4): Groq LLM initialized.")
    except Exception as e:
        print(f"--- set_env.py (v4) ERROR initializing Groq LLM: {e}")

embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("--- set_env.py (v4): HuggingFace Embeddings model loaded ('sentence-transformers/all-mpnet-base-v2').")
except Exception as e:
    print(f"--- set_env.py (v4) ERROR initializing HuggingFace Embeddings: {e}")


# === Qdrant Connection ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
print(f"--- set_env.py (v4): Attempting to connect to Qdrant. URL: {QDRANT_URL}, API Key Provided: {'Yes' if QDRANT_API_KEY else 'No'}")

client = None
if not QDRANT_URL:
    print("--- set_env.py (v4) ERROR: QDRANT_URL environment variable not set. Cannot connect to Qdrant.")
elif not QDRANT_URL.startswith("https://") and QDRANT_API_KEY :
    print(f"--- set_env.py (v4) WARNING: QDRANT_URL '{QDRANT_URL}' is not secure (does not start with https://) but an API key is provided. This is insecure.")
    if QDRANT_URL.startswith("http://"):
         try:
            print(f"--- set_env.py (v4): Attempting connection to insecure Qdrant at {QDRANT_URL} (no API key check here).")
            client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
            client.get_collections()
            print(f"--- set_env.py (v4): Successfully connected to insecure Qdrant at {QDRANT_URL} and listed collections.")
         except Exception as e:
            print(f"--- set_env.py (v4) ERROR connecting to insecure Qdrant at {QDRANT_URL}: {e}")
            client = None
    else:
        client = None
elif not QDRANT_URL.startswith("https://") and not QDRANT_API_KEY:
    try:
        print(f"--- set_env.py (v4): Connecting to Qdrant at {QDRANT_URL} (no API key, assuming local HTTP).")
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        client.get_collections()
        print(f"--- set_env.py (v4): Successfully connected to Qdrant at {QDRANT_URL} and listed collections.")
    except Exception as e:
        print(f"--- set_env.py (v4) ERROR connecting to Qdrant at {QDRANT_URL}: {e}")
        client = None
else: 
    try:
        if QDRANT_API_KEY:
            print(f"--- set_env.py (v4): Connecting to Qdrant Cloud at {QDRANT_URL} with API key.")
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            print(f"--- set_env.py (v4): Connecting to Qdrant Cloud at {QDRANT_URL} (no API key provided, HTTPS).")
            client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        print(f"--- set_env.py (v4): Successfully connected to Qdrant Cloud at {QDRANT_URL} and listed collections.")
    except Exception as e:
        print(f"--- set_env.py (v4) ERROR connecting to Qdrant Cloud at {QDRANT_URL}: {e}")
        client = None

collection_name = "allocations_docs"
uploaded_file_path = "Allocations.pdf" # This will be /app/Allocations.pdf inside the container
print(f"--- set_env.py (v4): PDF path for indexing is set to: {uploaded_file_path} (expected at /app/{uploaded_file_path} in container)")

FORCE_REINDEX_STR = os.getenv("FORCE_REINDEX", "False")
FORCE_REINDEX = FORCE_REINDEX_STR.lower() in ['true', '1', 't']
print(f"--- set_env.py (v4): FORCE_REINDEX set to: {FORCE_REINDEX}")

vector_store = None
if client and embeddings:
    print("--- set_env.py (v4): Qdrant client and embeddings model are available. Proceeding with collection check/setup.")
    collection_exists = False
    try:
        print(f"--- set_env.py (v4): Checking if collection '{collection_name}' exists...")
        client.get_collection(collection_name=collection_name)
        collection_exists = True
        print(f"--- set_env.py (v4): Collection '{collection_name}' found.")
    except Exception as e:
        if "not found" in str(e).lower() or "status_code=404" in str(e).lower() or "StatusCode.NOT_FOUND" in str(e).upper():
            print(f"--- set_env.py (v4): Collection '{collection_name}' does not exist. Will attempt to create.")
            collection_exists = False
        else:
            print(f"--- set_env.py (v4) Warning: Error trying to get collection '{collection_name}': {e}. Assuming it does not exist.")
            collection_exists = False

    if FORCE_REINDEX and collection_exists:
        try:
            print(f"--- set_env.py (v4): FORCE_REINDEX is True. Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            print(f"--- set_env.py (v4): Collection '{collection_name}' deleted successfully.")
            collection_exists = False
        except Exception as e:
            print(f"--- set_env.py (v4) ERROR deleting collection '{collection_name}': {e}")
            try:
                client.get_collection(collection_name=collection_name)
                print(f"--- set_env.py (v4): Collection '{collection_name}' still exists after failed delete attempt.")
                collection_exists = True
            except:
                collection_exists = False

    if not collection_exists:
        print(f"--- set_env.py (v4): Attempting to create and/or index collection '{collection_name}'...")
        # Path inside the container will be /app/Allocations.pdf
        container_pdf_path = os.path.join(os.getcwd(), uploaded_file_path) # os.getcwd() should be /app
        print(f"--- set_env.py (v4): Checking for PDF at absolute path inside container: {container_pdf_path}")
        if not os.path.exists(container_pdf_path): # Check existence of PDF at the expected path
            print(f"--- set_env.py (v4) CRITICAL ERROR: PDF file not found at '{container_pdf_path}'. Cannot index documents.")
        else:
            print(f"--- set_env.py (v4): PDF file found at '{container_pdf_path}'. Proceeding with loading and indexing.")
            try:
                vector_size_test_emb = embeddings.embed_query("test query for vector size")
                actual_vector_size = len(vector_size_test_emb)
                print(f"--- set_env.py (v4): Determined vector size: {actual_vector_size}")

                print(f"--- set_env.py (v4): Recreating collection '{collection_name}' with vector size {actual_vector_size}...")
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={"size": actual_vector_size, "distance": "Cosine"},
                )
                print(f"--- set_env.py (v4): Collection '{collection_name}' created/recreated successfully.")

                print("--- set_env.py (v4): Loading PDF with PyMuPDFLoader...")
                loader = PyMuPDFLoader(container_pdf_path, extract_images=False) # Use container_pdf_path
                docs = loader.load()
                print(f"--- set_env.py (v4): Loaded {len(docs)} pages from '{container_pdf_path}'.")

                print("--- set_env.py (v4): Splitting documents...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = splitter.split_documents(docs)
                print(f"--- set_env.py (v4): Created {len(chunked_docs)} document chunks.")
                
                if not chunked_docs:
                    print("--- set_env.py (v4) Warning: No document chunks were created. Check PDF content and splitter settings.")
                else:
                    print(f"--- set_env.py (v4): Attempting to add {len(chunked_docs)} chunks to Qdrant...")
                    temp_indexing_vector_store = QdrantVectorStore(
                        client=client,
                        collection_name=collection_name,
                        embeddings=embeddings,
                    )
                    temp_indexing_vector_store.add_documents(documents=chunked_docs)
                    print(f"--- set_env.py (v4): Successfully indexed {len(chunked_docs)} document chunks into '{collection_name}'.")
                collection_exists = True
            except Exception as e:
                print(f"--- set_env.py (v4) CRITICAL ERROR during document loading, processing, or indexing for '{collection_name}': {e}", exc_info=True)
                collection_exists = False
    else: 
         print(f"--- set_env.py (v4): Collection '{collection_name}' already exists and FORCE_REINDEX is False — skipping indexing.")

    if collection_exists:
        try:
            print(f"--- set_env.py (v4): Initializing query vector_store for collection '{collection_name}'...")
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings,
            )
            print("--- set_env.py (v4): Query vector_store initialized successfully.")
        except Exception as e:
            print(f"--- set_env.py (v4) ERROR initializing query vector_store for '{collection_name}': {e}")
            vector_store = None
    else:
        print(f"--- set_env.py (v4): Query vector_store NOT initialized because collection '{collection_name}' could not be confirmed or created/indexed.")
        vector_store = None
elif not client:
    print("--- set_env.py (v4): Qdrant client not initialized. Vector store will not be available.")
    vector_store = None
elif not embeddings:
    print("--- set_env.py (v4): Embeddings model not initialized. Vector store will not be available.")
    vector_store = None

if vector_store:
    print("--- set_env.py (v4) FINISHED: vector_store IS INITIALIZED ---")
else:
    print("--- set_env.py (v4) FINISHED: vector_store IS NONE. CHECK LOGS ABOVE FOR ERRORS. ---")

