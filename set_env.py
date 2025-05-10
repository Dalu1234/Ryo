import os
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["USER_AGENT"] = "RyoRAGBot/1.0"

# === Set Environment Variables ===
unique_id = uuid4().hex[:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"  # replace with real
os.environ["GROQ_API_KEY"] = "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"              # replace with real

# === Initialize LLM and Embeddings ===
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# === Connect to Qdrant ===
# new: connect over HTTP to your Docker container
client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=False      # disable gRPC if you run into HTTP vs gRPC issues
)
collection_name = "allocations_docs" # Ensure this is consistent
uploaded_file_path = "C:\\Users\\chukw\\Downloads\\Allocations.pdf"

# --- CONDITIONAL DELETION AND INDEXING ---
# Set this to True if you want to force re-indexing (e.g., after changing chunk_size or PDF)
# Set to False for normal operation where you just want to query an existing index.
FORCE_REINDEX = False # CHANGE THIS AS NEEDED

if FORCE_REINDEX:
    try:
        if client.collection_exists(collection_name):
            print(f"FORCE_REINDEX is True. Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
            print(f"Collection {collection_name} deleted.")
    except Exception as e:
        print(f"Error deleting collection: {e}")

# === Create collection and index documents IF IT DOESN'T EXIST (or was just deleted by FORCE_REINDEX) ===
if not client.collection_exists(collection_name):
    print("Collection not found or was deleted — indexing documents now...")
    dummy_vector = embeddings.embed_query("test") # Ensure this uses the same embeddings model
    vector_size = len(dummy_vector)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={"size": vector_size, "distance": "Cosine"},
    )

    # Load and split the PDF
    loader = PyMuPDFLoader(uploaded_file_path, extract_images=False)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")
    for doc in docs[:2]:
        print("Page metadata:", doc.metadata)
        print("Excerpt:", doc.page_content[:200], "…")

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    #all_splits = text_splitter.split_documents(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_docs = loader.load_and_split(splitter)
    print(f"Created {len(chunked_docs)} chunks.")

    # Index chunks
    # This specific QdrantVectorStore instance is for adding documents
    indexing_vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings, # Use the HuggingFace embeddings
    )
    indexing_vector_store.add_documents(documents=chunked_docs)
    print(f"Indexed {len(chunked_docs)} document chunks.")
else:
    print(f"Collection '{collection_name}' already exists — skipping indexing.")

# === Always define vector_store for querying ===
# This QdrantVectorStore instance is for querying (used by main.py)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings, # Use the HuggingFace embeddings
)

print("set_env.py finished. Vector store should be ready.")