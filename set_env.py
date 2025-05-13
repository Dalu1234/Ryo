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

# === Set General Environment Variables ===
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RyoRAGBot/1.0_Cloud")

# === LLM and Embeddings Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY environment variable not set. LLM will not function.")
    llm = None
else:
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3, groq_api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("HuggingFace Embeddings model loaded ('sentence-transformers/all-mpnet-base-v2').")


# === Qdrant Connection ===
# CRITICAL: Ensure QDRANT_URL in Render environment variables is your HTTPS Qdrant Cloud URL
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = None
if not QDRANT_URL:
    print("ERROR: QDRANT_URL environment variable not set. Cannot connect to Qdrant.")
elif not QDRANT_URL.startswith("https://"):
    print(f"ERROR: QDRANT_URL '{QDRANT_URL}' is not secure (does not start with https://). API key will not be sent securely. Please use your Qdrant Cloud HTTPS URL.")
    # Optionally, you could still try to connect if it's a local non-cloud http URL without an API key
    if not QDRANT_API_KEY and QDRANT_URL.startswith("http://"): # Allow local http if no API key
         try:
            print(f"Connecting to local Qdrant at {QDRANT_URL} (no API key, HTTP).")
            client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
            # Attempt a simple operation to confirm connectivity instead of health_check
            client.get_collections() 
            print("Successfully connected to local Qdrant and listed collections.")
         except Exception as e:
            print(f"ERROR connecting to local Qdrant at {QDRANT_URL}: {e}")
            client = None
    else: # If it's http and has an API key, or other misconfiguration
        client = None # Do not proceed with insecure API key usage to a non-https URL
else: # QDRANT_URL starts with https://
    try:
        if QDRANT_API_KEY:
            print(f"Connecting to Qdrant Cloud at {QDRANT_URL} with API key.")
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                # prefer_grpc=False # gRPC might be preferred for cloud if network allows
            )
        else:
            # This case (HTTPS Qdrant Cloud URL without API key) might be less common
            # but supported if the cluster is configured that way.
            print(f"Connecting to Qdrant Cloud at {QDRANT_URL} (no API key provided, HTTPS).")
            client = QdrantClient(
                url=QDRANT_URL,
                # prefer_grpc=False
            )
        # Attempt a simple operation to confirm connectivity instead of health_check
        client.get_collections() 
        print("Successfully connected to Qdrant Cloud and listed collections.")
    except Exception as e:
        print(f"ERROR connecting to Qdrant Cloud at {QDRANT_URL}: {e}")
        client = None

collection_name = "allocations_docs"
uploaded_file_path = "Allocations.pdf" # Ensure this file is in your Git repo

FORCE_REINDEX_STR = os.getenv("FORCE_REINDEX", "False")
FORCE_REINDEX = FORCE_REINDEX_STR.lower() in ['true', '1', 't']

vector_store = None
if client:
    collection_exists = False
    try:
        # Check if collection exists by trying to get its info
        client.get_collection(collection_name=collection_name)
        collection_exists = True
        print(f"Collection '{collection_name}' found.")
    except Exception as e: # Catches exceptions if collection does not exist or other connection issues
        # Check if the error message indicates "Not found" or similar
        if "not found" in str(e).lower() or "status_code=404" in str(e).lower():
            print(f"Collection '{collection_name}' does not exist. Will attempt to create.")
            collection_exists = False
        else:
            print(f"Error checking if collection '{collection_name}' exists: {e}. Assuming it does not exist.")
            collection_exists = False


    if FORCE_REINDEX and collection_exists:
        try:
            print(f"FORCE_REINDEX is True. Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            print(f"Collection {collection_name} deleted.")
            collection_exists = False
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")

    if not collection_exists:
        print(f"Attempting to create and/or index collection '{collection_name}'...")
        if not os.path.exists(uploaded_file_path):
            print(f"ERROR: PDF file not found at '{uploaded_file_path}'. Cannot index documents.")
        else:
            try:
                dummy_vector = embeddings.embed_query("test")
                vector_size = len(dummy_vector)

                client.recreate_collection( # Use recreate_collection for simplicity
                    collection_name=collection_name,
                    vectors_config={"size": vector_size, "distance": "Cosine"},
                )
                print(f"Collection '{collection_name}' created/recreated.")

                loader = PyMuPDFLoader(uploaded_file_path, extract_images=False)
                docs = loader.load()
                print(f"Loaded {len(docs)} pages from '{uploaded_file_path}'.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = splitter.split_documents(docs)
                print(f"Created {len(chunked_docs)} document chunks.")
                
                temp_indexing_vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embeddings=embeddings,
                )
                temp_indexing_vector_store.add_documents(documents=chunked_docs)
                print(f"Indexed {len(chunked_docs)} document chunks into '{collection_name}'.")
                collection_exists = True
            except Exception as e:
                print(f"ERROR during document indexing for '{collection_name}': {e}")
    else:
        if not FORCE_REINDEX: # Only print this if we weren't force re-indexing
             print(f"Collection '{collection_name}' already exists and FORCE_REINDEX is false — skipping indexing.")


    if collection_exists:
        try:
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
        print(f"Query vector_store not initialized because collection '{collection_name}' could not be confirmed or created.")
        vector_store = None
else:
    print("Qdrant client not initialized. Vector store will not be available.")
    vector_store = None

print("set_env.py finished.")

# --- Environment variables to set in Render: ---
# PYTHON_VERSION="3.10.13" (Example)
# LANGCHAIN_API_KEY="lsv2_..."
# GROQ_API_KEY="gsk_..."
# QDRANT_URL="YOUR_HTTPS_QDRANT_CLOUD_URL" (e.g., https://your-cluster.region.qdrant.cloud:6333)
# QDRANT_API_KEY="Your Qdrant Cloud API Key"
# FORCE_REINDEX="False"
# USER_AGENT="RyoRAGBot/1.0_Render"
