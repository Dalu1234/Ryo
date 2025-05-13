# main.py
from langchain import hub
# from langchain_core.documents import Document # Not directly used here
# from langchain_core.prompts import PromptTemplate # Not directly used here
from langgraph.graph import START, StateGraph # Make sure StateGraph is imported
from typing import List, TypedDict, Optional
from langchain_core.documents import Document # Ensure Document is imported

# llm and vector_store are now imported after potential environment variable setup
# and Qdrant connection attempt.
try:
    from set_env import llm, vector_store
    if llm is None:
        print("Error: LLM not initialized in set_env.py")
    if vector_store is None:
        print("Error: vector_store not initialized in set_env.py. RAG will not function.")
except ImportError:
    print("Error: Could not import from set_env.py. Make sure it exists and is configured.")
    llm = None
    vector_store = None
except Exception as e:
    print(f"An unexpected error occurred during imports from set_env: {e}")
    llm = None
    vector_store = None


# Use a prompt template from Langchain hub or define your own
# Ensure this is accessible, or define it directly if hub.pull fails in some environments
try:
    prompt = hub.pull("rlm/rag-prompt")
except Exception as e:
    print(f"Warning: Could not pull 'rlm/rag-prompt' from hub: {e}. Using a default prompt.")
    from langchain_core.prompts import ChatPromptTemplate
    # A generic fallback prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question: \n\n{context}"),
        ("human", "{question}")
    ])


# Define application state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    error: Optional[str] # Add an error field

# Retrieval step
def retrieve(state: State):
    if vector_store is None:
        print("Retrieve error: vector_store is not available.")
        return {"context": [], "error": "Vector store not available."}
    try:
        retrieved_docs = vector_store.similarity_search(state["question"], k=5) # Reduced k for free tier
        return {"context": retrieved_docs, "error": None}
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"context": [], "error": f"Retrieval failed: {e}"}

# Generation step
def generate(state: State):
    if state.get("error"): # If retrieval failed, pass the error
        return {"answer": state["error"]}
    if not state["context"]:
         return {"answer": "No relevant context found to answer the question."}
    if llm is None:
        print("Generate error: LLM is not available.")
        return {"answer": "LLM not available."}

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    try:
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"answer": f"Generation failed: {e}"}

# Build and run the graph
if llm and vector_store: # Only build graph if core components are ready
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()
    print("RAG graph compiled successfully.")
else:
    print("RAG graph could not be compiled due to missing LLM or vector_store.")
    graph = None

# Example invocation (primarily for local testing, app.py will handle invocations)
# if graph:
#     try:
#         result = graph.invoke({"question": "is the student activity fee collected from students solely enrolled in the College of General Studies?"})
#         print(f'Context: {result.get("context")}\n\n') # Use .get for safety
#         print(f'Answer: {result.get("answer")}')
#     except Exception as e:
#         print(f"Error invoking graph for test question: {e}")
# else:
#     print("Graph not available for test invocation.")