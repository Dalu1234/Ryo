# main.py (Debug Retrieval v2 - LangGraph Fix)
import sys
print("--- main.py (Debug Retrieval v2) IMPORTING ---"); sys.stdout.flush()

from langchain import hub
from langchain_core.documents import Document 
from langgraph.graph import START, StateGraph, END # Ensure END is imported
from typing import List, TypedDict, Optional 

# Import llm and vector_store from set_env
try:
    from set_env import llm, vector_store
    if llm is None:
        print("--- main.py (Debug Retrieval v2) ERROR: LLM not initialized in set_env.py"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (Debug Retrieval v2) ERROR: vector_store not initialized in set_env.py. RAG will not function."); sys.stdout.flush()
    else:
        print("--- main.py (Debug Retrieval v2): Successfully imported llm and vector_store."); sys.stdout.flush()
except ImportError as e:
    print(f"--- main.py (Debug Retrieval v2) ERROR: Could not import from set_env.py: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None
except Exception as e:
    print(f"--- main.py (Debug Retrieval v2) An unexpected error occurred during imports from set_env: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None

# Use a prompt template from Langchain hub or define your own
try:
    prompt = hub.pull("rlm/rag-prompt")
    print("--- main.py (Debug Retrieval v2): Successfully pulled 'rlm/rag-prompt' from hub."); sys.stdout.flush()
except Exception as e:
    print(f"--- main.py (Debug Retrieval v2) Warning: Could not pull 'rlm/rag-prompt' from hub: {e}. Using a default prompt."); sys.stdout.flush()
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
    error: Optional[str] 

# Retrieval step
def retrieve(state: State):
    print(f"--- main.py (Debug Retrieval v2) RETRIEVE node executing for question: '{state['question']}' ---"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (Debug Retrieval v2) RETRIEVE ERROR: vector_store is None."); sys.stdout.flush()
        return {"context": [], "error": "Vector store not available."} 
    
    retrieved_docs = [] 
    try:
        num_docs_to_retrieve = 5 
        print(f"--- main.py (Debug Retrieval v2) RETRIEVE: Performing similarity search for '{state['question']}' with k={num_docs_to_retrieve}"); sys.stdout.flush()
        retrieved_docs = vector_store.similarity_search(state["question"], k=num_docs_to_retrieve)
        
        print(f"--- main.py (Debug Retrieval v2) RETRIEVE: Found {len(retrieved_docs)} documents."); sys.stdout.flush()
        if not retrieved_docs:
            print("--- main.py (Debug Retrieval v2) RETRIEVE: No documents found by similarity search."); sys.stdout.flush()
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- main.py (Debug Retrieval v2) RETRIEVE: Doc {i+1} Metadata: {doc.metadata}"); sys.stdout.flush()
                print(f"--- main.py (Debug Retrieval v2) RETRIEVE: Doc {i+1} Content Snippet: {doc.page_content[:200]}..."); sys.stdout.flush()
        
        return {"context": retrieved_docs, "error": None} 
    except Exception as e:
        print(f"--- main.py (Debug Retrieval v2) RETRIEVE EXCEPTION: {e}", exc_info=True); sys.stdout.flush()
        return {"context": [], "error": f"Retrieval failed: {e}"}


# Generation step
def generate(state: State):
    print(f"--- main.py (Debug Retrieval v2) GENERATE node executing for question: '{state['question']}' ---"); sys.stdout.flush()
    
    if state.get("error"): 
        print(f"--- main.py (Debug Retrieval v2) GENERATE: Error from retrieval: {state['error']}"); sys.stdout.flush()
        return {"answer": state['error']}

    # Check if context is empty. If so, provide a specific message.
    # The 'rlm/rag-prompt' also has logic for this, but an explicit check is good.
    if not state.get("context"): 
        print("--- main.py (Debug Retrieval v2) GENERATE: Context is empty. Replying that no relevant context was found."); sys.stdout.flush()
        return {"answer": "No relevant context was found in the document to answer your question."}

    if llm is None:
        print("--- main.py (Debug Retrieval v2) GENERATE ERROR: LLM is None."); sys.stdout.flush()
        return {"answer": "LLM not available."}

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    print(f"--- main.py (Debug Retrieval v2) GENERATE: Content passed to LLM (first 300 chars): {docs_content[:300]}..."); sys.stdout.flush()

    try:
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        print("--- main.py (Debug Retrieval v2) GENERATE: Invoking LLM..."); sys.stdout.flush()
        response = llm.invoke(messages)
        print(f"--- main.py (Debug Retrieval v2) GENERATE: LLM response content: {response.content}"); sys.stdout.flush()
        return {"answer": response.content}
    except Exception as e:
        print(f"--- main.py (Debug Retrieval v2) GENERATE EXCEPTION: {e}", exc_info=True); sys.stdout.flush()
        return {"answer": f"LLM generation failed: {e}"}

# Build and run the graph
graph = None
if llm and vector_store: 
    print("--- main.py (Debug Retrieval v2): Compiling RAG graph..."); sys.stdout.flush()
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    graph_builder.set_entry_point("retrieve") 
    graph_builder.add_edge("retrieve", "generate")
    # graph_builder.set_finish_node("generate") # REMOVED This was causing the error
    graph_builder.add_edge("generate", END) # ADDED: Explicitly connect the last node to END

    graph = graph_builder.compile()
    print("--- main.py (Debug Retrieval v2): RAG graph compiled successfully."); sys.stdout.flush()
else:
    print("--- main.py (Debug Retrieval v2) ERROR: RAG graph could not be compiled due to missing LLM or vector_store."); sys.stdout.flush()

