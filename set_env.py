# set_env.py
import os
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader
# from dotenv import load_dotenv # Uncomment for local .env loading if you add it here

# load_dotenv() # Uncomment for local .env loading: call this before any os.getenv

print("--- Starting set_env.py execution ---")

# === Set General Environment Variables ===
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RyoRAGBot/1.0_Cloud")
print(f"USER_AGENT set to: {os.environ['USER_AGENT']}")

# === Langchain Tracing (Optional but good practice) ===
LANGCHAIN_TRACING_V2_ENV = os.getenv("LANGCHAIN_TRACING_V2", "false") # Default to false if not set
if LANGCHAIN_TRACING_V2_ENV.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", f"RyoRAG_AWS_{uuid4().hex[:8]}")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY_CONFIG = os.getenv("LANGCHAIN_API_KEY")
    if LANGCHAIN_API_KEY_CONFIG:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY_CONFIG
        print("LangSmith tracing configured.")
    else:
        print("Warning: LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set in environment.")
else:
    print("LangSmith tracing is disabled via LANGCHAIN_TRACING_V2 environment variable.")


# === LLM and Embeddings Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = None
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY environment variable not set. LLM will not function.")
else:
    try:
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3, groq_api_key=GROQ_API_KEY)
        print("Groq LLM initialized.")
    except Exception as e:
        print(f"ERROR initializing Groq LLM: {e}")

embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("HuggingFace Embeddings model loaded ('sentence-transformers/all-mpnet-base-v2').")
except Exception as e:
    print(f"ERROR initializing HuggingFace Embeddings: {e}")


# === Qdrant Connection ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
print(f"Attempting to connect to Qdrant. URL: {QDRANT_URL}, API Key Provided: {'Yes' if QDRANT_API_KEY else 'No'}")

client = None
if not QDRANT_URL:
    print("ERROR: QDRANT_URL environment variable not set. Cannot connect to Qdrant.")
elif not QDRANT_URL.startswith("https://") and QDRANT_API_KEY : # Warn if API key used with HTTP
    print(f"WARNING: QDRANT_URL '{QDRANT_URL}' is not secure (does not start with https://) but an API key is provided. This is insecure.")
    # Forcing client to None if API key is present with non-HTTPS URL for security.
    # If it's a local http URL without an API key, it might be fine for testing.
    if QDRANT_URL.startswith("http://"):
         try:
            print(f"Attempting connection to insecure Qdrant at {QDRANT_URL} (no API key check here).")
            client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
            client.get_collections()
            print(f"Successfully connected to insecure Qdrant at {QDRANT_URL} and listed collections.")
         except Exception as e:
            print(f"ERROR connecting to insecure Qdrant at {QDRANT_URL}: {e}")
            client = None
    else:
        client = None
elif not QDRANT_URL.startswith("https://") and not QDRANT_API_KEY: # HTTP without API key (e.g. local)
    try:
        print(f"Connecting to Qdrant at {QDRANT_URL} (no API key, assuming local HTTP).")
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        client.get_collections()
        print(f"Successfully connected to Qdrant at {QDRANT_URL} and listed collections.")
    except Exception as e:
        print(f"ERROR connecting to Qdrant at {QDRANT_URL}: {e}")
        client = None
else: # QDRANT_URL starts with https://
    try:
        if QDRANT_API_KEY:
            print(f"Connecting to Qdrant Cloud at {QDRANT_URL} with API key.")
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            print(f"Connecting to Qdrant Cloud at {QDRANT_URL} (no API key provided, HTTPS).")
            client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        print(f"Successfully connected to Qdrant Cloud at {QDRANT_URL} and listed collections.")
    except Exception as e:
        print(f"ERROR connecting to Qdrant Cloud at {QDRANT_URL}: {e}")
        client = None

collection_name = "allocations_docs"
# CRITICAL: Ensure 'Allocations.pdf' is in your Git repo (root or adjust path as needed).
# This path is relative to the root of your project where the script runs.
uploaded_file_path = "Allocations.pdf"
print(f"Looking for PDF at: {os.path.abspath(uploaded_file_path)}") # Show absolute path for debugging

FORCE_REINDEX_STR = os.getenv("FORCE_REINDEX", "False")
FORCE_REINDEX = FORCE_REINDEX_STR.lower() in ['true', '1', 't']
print(f"FORCE_REINDEX set to: {FORCE_REINDEX}")

vector_store = None
if client and embeddings: # Only proceed if Qdrant client AND embeddings are initialized
    print("Qdrant client and embeddings model are available. Proceeding with collection check/setup.")
    collection_exists = False
    try:
        print(f"Checking if collection '{collection_name}' exists...")
        client.get_collection(collection_name=collection_name)
        collection_exists = True
        print(f"Collection '{collection_name}' found.")
    except Exception as e:
        if "not found" in str(e).lower() or "status_code=404" in str(e).lower() or "StatusCode.NOT_FOUND" in str(e):
            print(f"Collection '{collection_name}' does not exist. Will attempt to create.")
            collection_exists = False
        else:
            print(f"Warning: Error trying to get collection '{collection_name}': {e}. Assuming it does not exist.")
            collection_exists = False

    if FORCE_REINDEX and collection_exists:
        try:
            print(f"FORCE_REINDEX is True. Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
            collection_exists = False
        except Exception as e:
            print(f"ERROR deleting collection '{collection_name}': {e}")
            # If deletion fails, we might not want to proceed with recreation if it still exists
            # Re-check existence
            try:
                client.get_collection(collection_name=collection_name)
                print(f"Collection '{collection_name}' still exists after failed delete attempt.")
                collection_exists = True # It's still there
            except:
                collection_exists = False # Assume it's gone or was never there


    if not collection_exists:
        print(f"Attempting to create and/or index collection '{collection_name}'...")
        if not os.path.exists(uploaded_file_path):
            print(f"CRITICAL ERROR: PDF file not found at '{uploaded_file_path}'. Cannot index documents.")
        else:
            print(f"PDF file found at '{uploaded_file_path}'. Proceeding with loading and indexing.")
            try:
                vector_size = embeddings.embed_query("test query for vector size") # Get vector size from actual model
                print(f"Determined vector size: {len(vector_size)}")

                print(f"Recreating collection '{collection_name}'...")
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={"size": len(vector_size), "distance": "Cosine"},
                )
                print(f"Collection '{collection_name}' created/recreated successfully.")

                print("Loading PDF with PyMuPDFLoader...")
                loader = PyMuPDFLoader(uploaded_file_path, extract_images=False)
                docs = loader.load()
                print(f"Loaded {len(docs)} pages from '{uploaded_file_path}'.")

                print("Splitting documents...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = splitter.split_documents(docs)
                print(f"Created {len(chunked_docs)} document chunks.")
                
                if not chunked_docs:
                    print("Warning: No document chunks were created. Check PDF content and splitter settings.")
                else:
                    print(f"Attempting to add {len(chunked_docs)} chunks to Qdrant...")
                    # Directly use QdrantClient for adding points if QdrantVectorStore causes issues during setup
                    # Or ensure QdrantVectorStore is robust
                    temp_indexing_vector_store = QdrantVectorStore(
                        client=client,
                        collection_name=collection_name,
                        embeddings=embeddings,
                    )
                    temp_indexing_vector_store.add_documents(documents=chunked_docs)
                    print(f"Successfully indexed {len(chunked_docs)} document chunks into '{collection_name}'.")
                collection_exists = True # Mark as existing after successful creation and indexing
            except Exception as e:
                print(f"CRITICAL ERROR during document loading, processing, or indexing for '{collection_name}': {e}", exc_info=True)
                collection_exists = False # Explicitly mark as false if indexing fails
    else: # Collection exists and FORCE_REINDEX is False
         print(f"Collection '{collection_name}' already exists and FORCE_REINDEX is False — skipping indexing.")

    # Initialize vector_store for querying if collection exists
    if collection_exists:
        try:
            print(f"Initializing query vector_store for collection '{collection_name}'...")
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings,
            )
            print("Query vector_store initialized successfully.")
        except Exception as e:
            print(f"ERROR initializing query vector_store for '{collection_name}': {e}")
            vector_store = None
    else:
        print(f"Query vector_store NOT initialized because collection '{collection_name}' could not be confirmed or created/indexed.")
        vector_store = None
elif not client:
    print("Qdrant client not initialized. Vector store will not be available.")
    vector_store = None
elif not embeddings:
    print("Embeddings model not initialized. Vector store will not be available.")
    vector_store = None


if vector_store:
    print("--- set_env.py finished: vector_store IS INITIALIZED ---")
else:
    print("--- set_env.py finished: vector_store IS NONE ---")

