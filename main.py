# main.py (Debug Retrieval v1 - for non-conversational graph)
import sys
print("--- main.py (Debug Retrieval v1) IMPORTING ---"); sys.stdout.flush()

from langchain import hub
from langchain_core.documents import Document # Ensure Document is imported
# from langchain_core.prompts import PromptTemplate # Not directly used if hub.pull works
from langgraph.graph import START, StateGraph # Make sure StateGraph is imported
from typing import List, TypedDict, Optional # Added Optional

# Import llm and vector_store from set_env
try:
    from set_env import llm, vector_store
    if llm is None:
        print("--- main.py (Debug Retrieval v1) ERROR: LLM not initialized in set_env.py"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (Debug Retrieval v1) ERROR: vector_store not initialized in set_env.py. RAG will not function."); sys.stdout.flush()
    else:
        print("--- main.py (Debug Retrieval v1): Successfully imported llm and vector_store."); sys.stdout.flush()
except ImportError as e:
    print(f"--- main.py (Debug Retrieval v1) ERROR: Could not import from set_env.py: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None
except Exception as e:
    print(f"--- main.py (Debug Retrieval v1) An unexpected error occurred during imports from set_env: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None

# Use a prompt template from Langchain hub or define your own
try:
    prompt = hub.pull("rlm/rag-prompt")
    print("--- main.py (Debug Retrieval v1): Successfully pulled 'rlm/rag-prompt' from hub."); sys.stdout.flush()
except Exception as e:
    print(f"--- main.py (Debug Retrieval v1) Warning: Could not pull 'rlm/rag-prompt' from hub: {e}. Using a default prompt."); sys.stdout.flush()
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question. If the context is not relevant or doesn't provide an answer, say that you don't know or cannot find the answer in the provided material.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

# Define application state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    error: Optional[str] # Added for consistency, though not used in this simpler graph version

# Retrieval step
def retrieve(state: State):
    print(f"--- main.py (Debug Retrieval v1) RETRIEVE node executing for question: '{state['question']}' ---"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (Debug Retrieval v1) RETRIEVE ERROR: vector_store is None."); sys.stdout.flush()
        # Return empty context so generate node can handle it
        return {"context": [], "error": "Vector store not available."} 
    
    retrieved_docs = [] # Initialize to empty list
    try:
        # Using k=5 for a balance. Your original was k=20.
        # If memory/performance is an issue with k=20, k=5 is a reasonable start.
        num_docs_to_retrieve = 5 
        print(f"--- main.py (Debug Retrieval v1) RETRIEVE: Performing similarity search for '{state['question']}' with k={num_docs_to_retrieve}"); sys.stdout.flush()
        retrieved_docs = vector_store.similarity_search(state["question"], k=num_docs_to_retrieve)
        
        print(f"--- main.py (Debug Retrieval v1) RETRIEVE: Found {len(retrieved_docs)} documents."); sys.stdout.flush()
        if not retrieved_docs:
            print("--- main.py (Debug Retrieval v1) RETRIEVE: No documents found by similarity search."); sys.stdout.flush()
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- main.py (Debug Retrieval v1) RETRIEVE: Doc {i+1} Metadata: {doc.metadata}"); sys.stdout.flush()
                print(f"--- main.py (Debug Retrieval v1) RETRIEVE: Doc {i+1} Content Snippet: {doc.page_content[:200]}..."); sys.stdout.flush()
        
        return {"context": retrieved_docs, "error": None} # Pass error as None if successful
    except Exception as e:
        print(f"--- main.py (Debug Retrieval v1) RETRIEVE EXCEPTION: {e}", exc_info=True); sys.stdout.flush()
        return {"context": [], "error": f"Retrieval failed: {e}"}


# Generation step
def generate(state: State):
    print(f"--- main.py (Debug Retrieval v1) GENERATE node executing for question: '{state['question']}' ---"); sys.stdout.flush()
    
    if state.get("error"): # Check if retrieval passed an error
        print(f"--- main.py (Debug Retrieval v1) GENERATE: Error from retrieval: {state['error']}"); sys.stdout.flush()
        return {"answer": state['error']}

    if not state.get("context"): # Check if context is empty or None
        print("--- main.py (Debug Retrieval v1) GENERATE: Context is empty. Replying that no relevant context was found."); sys.stdout.flush()
        return {"answer": "No relevant context was found in the document to answer your question."}

    if llm is None:
        print("--- main.py (Debug Retrieval v1) GENERATE ERROR: LLM is None."); sys.stdout.flush()
        return {"answer": "LLM not available."}

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    print(f"--- main.py (Debug Retrieval v1) GENERATE: Content passed to LLM (first 300 chars): {docs_content[:300]}..."); sys.stdout.flush()

    try:
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        print("--- main.py (Debug Retrieval v1) GENERATE: Invoking LLM..."); sys.stdout.flush()
        response = llm.invoke(messages)
        print(f"--- main.py (Debug Retrieval v1) GENERATE: LLM response content: {response.content}"); sys.stdout.flush()
        return {"answer": response.content}
    except Exception as e:
        print(f"--- main.py (Debug Retrieval v1) GENERATE EXCEPTION: {e}", exc_info=True); sys.stdout.flush()
        return {"answer": f"LLM generation failed: {e}"}

# Build and run the graph
graph = None
if llm and vector_store: # Only build graph if core components are ready
    print("--- main.py (Debug Retrieval v1): Compiling RAG graph..."); sys.stdout.flush()
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    graph_builder.set_entry_point("retrieve") # Explicitly set entry point
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.set_finish_node("generate") # Explicitly set finish node

    graph = graph_builder.compile()
    print("--- main.py (Debug Retrieval v1): RAG graph compiled successfully."); sys.stdout.flush()
else:
    print("--- main.py (Debug Retrieval v1) ERROR: RAG graph could not be compiled due to missing LLM or vector_store."); sys.stdout.flush()

# The graph is now imported by app.py and invoked there.
# No need for example invocation here in a deployed app.
