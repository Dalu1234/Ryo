# set_env.py (v4.4 - Smaller Embedding Model)
import sys
print("--- EXECUTING set_env.py (v4.4) - Smaller Embedding Model ---"); sys.stdout.flush()
import os
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader

# --- Initial Debug Prints ---
print(f"--- set_env.py (v4.4) DEBUG: Python executable: {sys.executable}"); sys.stdout.flush()
print(f"--- set_env.py (v4.4) DEBUG: Current working directory: {os.getcwd()}"); sys.stdout.flush()
try:
    print(f"--- set_env.py (v4.4) DEBUG: Files in current directory ({os.getcwd()}): {os.listdir('.')}"); sys.stdout.flush()
    app_allocations_pdf_path = "/app/Allocations.pdf"
    if os.path.exists(app_allocations_pdf_path):
        print(f"--- set_env.py (v4.4) DEBUG: '{app_allocations_pdf_path}' exists inside the container."); sys.stdout.flush()
    else:
        relative_allocations_pdf_path = "Allocations.pdf"
        if os.path.exists(relative_allocations_pdf_path):
            print(f"--- set_env.py (v4.4) DEBUG: Relative path 'Allocations.pdf' exists in CWD '{os.getcwd()}'."); sys.stdout.flush()
        else:
            print(f"--- set_env.py (v4.4) DEBUG: Neither '{app_allocations_pdf_path}' nor relative 'Allocations.pdf' in CWD '{os.getcwd()}' found. Check Dockerfile COPY and PDF location."); sys.stdout.flush()
except Exception as e:
    print(f"--- set_env.py (v4.4) DEBUG: Error listing directory contents: {e}"); sys.stdout.flush()

# === Set General Environment Variables ===
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RyoRAGBot/1.0_Cloud")
print(f"--- set_env.py (v4.4): USER_AGENT set to: {os.environ['USER_AGENT']}"); sys.stdout.flush()

# === Langchain Tracing (Optional but good practice) ===
LANGCHAIN_TRACING_V2_ENV = os.getenv("LANGCHAIN_TRACING_V2", "false")
if LANGCHAIN_TRACING_V2_ENV.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", f"RyoRAG_AWS_{uuid4().hex[:8]}")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY_CONFIG = os.getenv("LANGCHAIN_API_KEY")
    if LANGCHAIN_API_KEY_CONFIG:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY_CONFIG
        print("--- set_env.py (v4.4): LangSmith tracing configured."); sys.stdout.flush()
    else:
        print("--- set_env.py (v4.4) Warning: LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set in environment."); sys.stdout.flush()
else:
    print("--- set_env.py (v4.4): LangSmith tracing is disabled (LANGCHAIN_TRACING_V2 is not 'true')."); sys.stdout.flush()

# === LLM and Embeddings Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = None
if not GROQ_API_KEY:
    print("--- set_env.py (v4.4) ERROR: GROQ_API_KEY environment variable not set. LLM will not function."); sys.stdout.flush()
else:
    try:
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3, groq_api_key=GROQ_API_KEY)
        print("--- set_env.py (v4.4): Groq LLM initialized."); sys.stdout.flush()
    except Exception as e:
        print(f"--- set_env.py (v4.4) ERROR initializing Groq LLM: {e}"); sys.stdout.flush()

embeddings_model = None
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2" # Using a smaller model
print(f"--- set_env.py (v4.4): Attempting to load HuggingFace Embeddings model: {embedding_model_name}"); sys.stdout.flush()
try:
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"--- set_env.py (v4.4): HuggingFace Embeddings model '{embedding_model_name}' loaded successfully."); sys.stdout.flush()
except Exception as e:
    print(f"--- set_env.py (v4.4) ERROR initializing HuggingFace Embeddings model '{embedding_model_name}': {e}"); sys.stdout.flush()

# === Qdrant Connection ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
print(f"--- set_env.py (v4.4): Attempting to connect to Qdrant. URL: {QDRANT_URL}, API Key Provided: {'Yes' if QDRANT_API_KEY else 'No'}"); sys.stdout.flush()

client = None
if not QDRANT_URL:
    print("--- set_env.py (v4.4) ERROR: QDRANT_URL environment variable not set. Cannot connect to Qdrant."); sys.stdout.flush()
elif not QDRANT_URL.startswith("https://") and QDRANT_API_KEY :
    print(f"--- set_env.py (v4.4) WARNING: QDRANT_URL '{QDRANT_URL}' is not secure (does not start with https://) but an API key is provided. This is insecure."); sys.stdout.flush()
    if QDRANT_URL.startswith("http://"):
         try:
            print(f"--- set_env.py (v4.4): Attempting connection to insecure Qdrant at {QDRANT_URL} (no API key check here)."); sys.stdout.flush()
            client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
            client.get_collections()
            print(f"--- set_env.py (v4.4): Successfully connected to insecure Qdrant at {QDRANT_URL} and listed collections."); sys.stdout.flush()
         except Exception as e:
            print(f"--- set_env.py (v4.4) ERROR connecting to insecure Qdrant at {QDRANT_URL}: {e}"); sys.stdout.flush()
            client = None
    else:
        client = None
elif not QDRANT_URL.startswith("https://") and not QDRANT_API_KEY:
    try:
        print(f"--- set_env.py (v4.4): Connecting to Qdrant at {QDRANT_URL} (no API key, assuming local HTTP)."); sys.stdout.flush()
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        client.get_collections()
        print(f"--- set_env.py (v4.4): Successfully connected to Qdrant at {QDRANT_URL} and listed collections."); sys.stdout.flush()
    except Exception as e:
        print(f"--- set_env.py (v4.4) ERROR connecting to Qdrant at {QDRANT_URL}: {e}"); sys.stdout.flush()
        client = None
else: 
    try:
        if QDRANT_API_KEY:
            print(f"--- set_env.py (v4.4): Connecting to Qdrant Cloud at {QDRANT_URL} with API key."); sys.stdout.flush()
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            print(f"--- set_env.py (v4.4): Connecting to Qdrant Cloud at {QDRANT_URL} (no API key provided, HTTPS)."); sys.stdout.flush()
            client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        print(f"--- set_env.py (v4.4): Successfully connected to Qdrant Cloud at {QDRANT_URL} and listed collections."); sys.stdout.flush()
    except Exception as e:
        print(f"--- set_env.py (v4.4) ERROR connecting to Qdrant Cloud at {QDRANT_URL}: {e}"); sys.stdout.flush()
        client = None

collection_name = "allocations_docs_minilm" # Changed collection name for the new model
uploaded_file_path = "Allocations.pdf" 
print(f"--- set_env.py (v4.4): PDF path for indexing is set to: {uploaded_file_path} (expected at /app/{uploaded_file_path} in container)"); sys.stdout.flush()
print(f"--- set_env.py (v4.4): Using Qdrant collection name: {collection_name}"); sys.stdout.flush()


FORCE_REINDEX_STR = os.getenv("FORCE_REINDEX", "False")
FORCE_REINDEX = FORCE_REINDEX_STR.lower() in ['true', '1', 't']
print(f"--- set_env.py (v4.4): FORCE_REINDEX set to: {FORCE_REINDEX}"); sys.stdout.flush()

vector_store = None
if client and embeddings_model: 
    print("--- set_env.py (v4.4): Qdrant client and embeddings model are available. Proceeding with collection check/setup."); sys.stdout.flush()
    collection_exists = False
    try:
        print(f"--- set_env.py (v4.4): Checking if collection '{collection_name}' exists..."); sys.stdout.flush()
        client.get_collection(collection_name=collection_name)
        collection_exists = True
        print(f"--- set_env.py (v4.4): Collection '{collection_name}' found."); sys.stdout.flush()
    except Exception as e:
        if "not found" in str(e).lower() or "status_code=404" in str(e).lower() or "StatusCode.NOT_FOUND" in str(e).upper(): # More robust check
            print(f"--- set_env.py (v4.4): Collection '{collection_name}' does not exist. Will attempt to create."); sys.stdout.flush()
            collection_exists = False
        else:
            print(f"--- set_env.py (v4.4) Warning: Error trying to get collection '{collection_name}': {e}. Assuming it does not exist."); sys.stdout.flush()
            collection_exists = False

    if FORCE_REINDEX and collection_exists:
        try:
            print(f"--- set_env.py (v4.4): FORCE_REINDEX is True. Deleting existing collection: {collection_name}"); sys.stdout.flush()
            client.delete_collection(collection_name=collection_name)
            print(f"--- set_env.py (v4.4): Collection '{collection_name}' deleted successfully."); sys.stdout.flush()
            collection_exists = False
        except Exception as e:
            print(f"--- set_env.py (v4.4) ERROR deleting collection '{collection_name}': {e}"); sys.stdout.flush()
            try:
                client.get_collection(collection_name=collection_name)
                print(f"--- set_env.py (v4.4): Collection '{collection_name}' still exists after failed delete attempt."); sys.stdout.flush()
                collection_exists = True
            except:
                collection_exists = False

    if not collection_exists:
        print(f"--- set_env.py (v4.4): Attempting to create and/or index collection '{collection_name}'..."); sys.stdout.flush()
        container_pdf_path = uploaded_file_path 
        print(f"--- set_env.py (v4.4): Checking for PDF at path inside container: '{container_pdf_path}' (relative to /app)"); sys.stdout.flush()
        if not os.path.exists(container_pdf_path): 
            print(f"--- set_env.py (v4.4) CRITICAL ERROR: PDF file not found at '{container_pdf_path}'. Cannot index documents."); sys.stdout.flush()
        else:
            print(f"--- set_env.py (v4.4): PDF file found at '{container_pdf_path}'. Proceeding with loading and indexing."); sys.stdout.flush()
            try:
                vector_size_test_emb = embeddings_model.embed_query("test query for vector size") 
                actual_vector_size = len(vector_size_test_emb)
                print(f"--- set_env.py (v4.4): Determined vector size for '{embedding_model_name}': {actual_vector_size}"); sys.stdout.flush() # Log model name

                print(f"--- set_env.py (v4.4): Recreating collection '{collection_name}' with vector size {actual_vector_size}..."); sys.stdout.flush()
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={"size": actual_vector_size, "distance": "Cosine"},
                )
                print(f"--- set_env.py (v4.4): Collection '{collection_name}' created/recreated successfully."); sys.stdout.flush()

                print("--- set_env.py (v4.4): Loading PDF with PyMuPDFLoader..."); sys.stdout.flush()
                loader = PyMuPDFLoader(container_pdf_path, extract_images=False)
                docs = loader.load()
                print(f"--- set_env.py (v4.4): Loaded {len(docs)} pages from '{container_pdf_path}'."); sys.stdout.flush()

                print("--- set_env.py (v4.4): Splitting documents..."); sys.stdout.flush()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = splitter.split_documents(docs)
                print(f"--- set_env.py (v4.4): Created {len(chunked_docs)} document chunks."); sys.stdout.flush()
                
                if not chunked_docs:
                    print("--- set_env.py (v4.4) Warning: No document chunks were created. Check PDF content and splitter settings."); sys.stdout.flush()
                else:
                    print(f"--- set_env.py (v4.4): Attempting to add {len(chunked_docs)} chunks to Qdrant collection '{collection_name}'..."); sys.stdout.flush()
                    temp_indexing_vector_store = QdrantVectorStore(
                        client=client,
                        collection_name=collection_name,
                        embedding=embeddings_model, 
                    )
                    temp_indexing_vector_store.add_documents(documents=chunked_docs)
                    print(f"--- set_env.py (v4.4): Successfully indexed {len(chunked_docs)} document chunks into '{collection_name}'."); sys.stdout.flush()
                collection_exists = True
            except Exception as e:
                print(f"--- set_env.py (v4.4) CRITICAL ERROR during document loading, processing, or indexing for '{collection_name}': {e}", exc_info=True); sys.stdout.flush()
                collection_exists = False
    else: 
         print(f"--- set_env.py (v4.4): Collection '{collection_name}' already exists and FORCE_REINDEX is False — skipping indexing."); sys.stdout.flush()

    if collection_exists:
        try:
            print(f"--- set_env.py (v4.4): Initializing query vector_store for collection '{collection_name}'..."); sys.stdout.flush()
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings_model, 
            )
            print("--- set_env.py (v4.4): Query vector_store initialized successfully."); sys.stdout.flush()
        except Exception as e:
            print(f"--- set_env.py (v4.4) ERROR initializing query vector_store for '{collection_name}': {e}", exc_info=True); sys.stdout.flush() 
            vector_store = None
    else:
        print(f"--- set_env.py (v4.4): Query vector_store NOT initialized because collection '{collection_name}' could not be confirmed or created/indexed."); sys.stdout.flush()
        vector_store = None
elif not client:
    print("--- set_env.py (v4.4): Qdrant client not initialized. Vector store will not be available."); sys.stdout.flush()
    vector_store = None
elif not embeddings_model: 
    print("--- set_env.py (v4.4): Embeddings model not initialized. Vector store will not be available."); sys.stdout.flush()
    vector_store = None

if vector_store:
    print("--- set_env.py (v4.4) FINISHED: vector_store IS INITIALIZED ---"); sys.stdout.flush()
else:
    print("--- set_env.py (v4.4) FINISHED: vector_store IS NONE. CHECK LOGS ABOVE FOR ERRORS. ---"); sys.stdout.flush()

