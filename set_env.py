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
# USER_AGENT is set by Render or can be a default
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RyoRAGBot/1.0_Cloud")

# === Langchain Tracing (Optional but good practice) ===
# These should be set directly in Render's environment variable settings if you use LangSmith.
# Example:
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
# LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", f"RyoRAG_Render_{uuid4().hex[:8]}")
# LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
# LANGCHAIN_API_KEY_CONFIG = os.getenv("LANGCHAIN_API_KEY") # Fetched for LangSmith

# if LANGCHAIN_TRACING_V2 == "true":
#     os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
#     os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
#     os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
#     if LANGCHAIN_API_KEY_CONFIG:
#         os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY_CONFIG
#     else:
#         print("Warning: LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set.")


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
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Qdrant Cloud might require an API key

client = None # Initialize client to None
if not QDRANT_URL:
    print("ERROR: QDRANT_URL environment variable not set. Cannot connect to Qdrant.")
else:
    try:
        if QDRANT_API_KEY:
            print(f"Connecting to Qdrant Cloud at {QDRANT_URL} with API key.")
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=False # For cloud, HTTP is often more straightforward
            )
        else:
            print(f"Connecting to Qdrant at {QDRANT_URL} (no API key provided).")
            client = QdrantClient(
                url=QDRANT_URL,
                prefer_grpc=False
            )
        client.health_check() # Verify connection
        print("Successfully connected to Qdrant.")
    except Exception as e:
        print(f"ERROR connecting to Qdrant: {e}")
        client = None # Ensure client is None on failure

collection_name = "allocations_docs"
# IMPORTANT: Ensure 'Allocations.pdf' is in your Git repo (root or adjust path).
uploaded_file_path = "Allocations.pdf"

# --- Conditional Re-indexing ---
FORCE_REINDEX_STR = os.getenv("FORCE_REINDEX", "False")
FORCE_REINDEX = FORCE_REINDEX_STR.lower() in ['true', '1', 't']

vector_store = None # Initialize vector_store to None
if client: # Only proceed if Qdrant client is initialized
    collection_exists = False
    try:
        collection_exists = client.collection_exists(collection_name=collection_name)
    except Exception as e:
        print(f"Error checking if collection '{collection_name}' exists: {e}")
        # client might be unhealthy or URL is wrong, treat as collection not existing for safety.
        collection_exists = False


    if FORCE_REINDEX:
        if collection_exists:
            try:
                print(f"FORCE_REINDEX is True. Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name=collection_name)
                print(f"Collection {collection_name} deleted.")
                collection_exists = False # Mark as not existing so it gets recreated
            except Exception as e:
                print(f"Error deleting collection '{collection_name}': {e}")
        else:
            print(f"FORCE_REINDEX is True, but collection '{collection_name}' does not exist. Will proceed to create and index.")


    if not collection_exists:
        print(f"Collection '{collection_name}' not found or was deleted — attempting to index documents now...")
        if not os.path.exists(uploaded_file_path):
            print(f"ERROR: PDF file not found at '{uploaded_file_path}'. Cannot index documents.")
        else:
            try:
                dummy_vector = embeddings.embed_query("test") # Test embedding model
                vector_size = len(dummy_vector)

                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"size": vector_size, "distance": "Cosine"},
                )
                print(f"Collection '{collection_name}' created.")

                loader = PyMuPDFLoader(uploaded_file_path, extract_images=False)
                docs = loader.load()
                print(f"Loaded {len(docs)} pages from '{uploaded_file_path}'.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = splitter.split_documents(docs)
                print(f"Created {len(chunked_docs)} document chunks.")

                # Index chunks
                # This QdrantVectorStore instance is for adding documents
                # It's fine to create it here for this specific task.
                temp_indexing_vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embeddings=embeddings,
                )
                temp_indexing_vector_store.add_documents(documents=chunked_docs)
                print(f"Indexed {len(chunked_docs)} document chunks into '{collection_name}'.")
                collection_exists = True # Collection should now exist
            except Exception as e:
                print(f"ERROR during document indexing for '{collection_name}': {e}")
    else:
        print(f"Collection '{collection_name}' already exists — skipping indexing.")

    # === Vector Store for Querying (used by main.py) ===
    # Only create the query vector_store if the collection is confirmed to exist or was just created
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
        print(f"Query vector_store not initialized because collection '{collection_name}' does not exist or indexing failed.")
        vector_store = None

    print("set_env.py finished.")
else:
    print("Qdrant client not initialized. Vector store will not be available.")
    vector_store = None # Ensure vector_store is defined as None if client is None

# --- Environment variables to set in Render: ---
# PYTHON_VERSION="3.10" (or your preferred version)
# LANGCHAIN_API_KEY="lsv2_..." (Your Langchain API Key for LangSmith, if used)
# GROQ_API_KEY="gsk_..." (Your Groq API Key)
# QDRANT_URL="Your Qdrant Cloud Cluster URL"
# QDRANT_API_KEY="Your Qdrant Cloud API Key" (if your Qdrant Cloud cluster uses one)
# FORCE_REINDEX="False" (Set to "True" only temporarily if you need to re-index on a specific deploy)
# USER_AGENT="RyoRAGBot/1.0_Render"
# Optional LangSmith variables:
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_PROJECT="Your RAG Project Name on LangSmith"
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
