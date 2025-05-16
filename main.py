# main.py (v2 - Conversational RAG with all-mpnet-base-v2 model)
import sys
print("--- main.py (v2 - Conversational RAG with mpnet) IMPORTING ---"); sys.stdout.flush()

from typing import List, Optional, Annotated, TypedDict
from langchain_core.documents import Document # Not directly used in this state, but good to have if needed
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage # BaseMessage not directly used
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate # For the final generation prompt

from langgraph.graph import StateGraph, END, MessagesState # Use MessagesState
from langgraph.prebuilt import ToolNode, tools_condition # For conditional tool execution
from langgraph.checkpoint.memory import MemorySaver # For in-memory chat history

# Import llm, vector_store, and embeddings_model from set_env
# embeddings_model is needed if we want to explicitly show its name or use it directly here,
# but QdrantVectorStore uses it internally.
# collection_name is also useful for clarity.
try:
    from set_env import llm, vector_store, embeddings_model, collection_name as qdrant_collection_name
    if llm is None:
        print("--- main.py (v2) ERROR: LLM not initialized in set_env.py"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (v2) ERROR: vector_store not initialized in set_env.py. RAG will not function."); sys.stdout.flush()
    if embeddings_model is None: # Check if embeddings_model itself was initialized
        print("--- main.py (v2) ERROR: embeddings_model not initialized in set_env.py."); sys.stdout.flush()
    else:
        print(f"--- main.py (v2): Successfully imported llm, vector_store (for collection '{qdrant_collection_name}'), and embeddings_model '{embeddings_model.model_name}'."); sys.stdout.flush()

except ImportError as e:
    print(f"--- main.py (v2) ERROR: Could not import from set_env.py: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None
    embeddings_model = None
    qdrant_collection_name = "unknown_collection"
except Exception as e:
    print(f"--- main.py (v2) An unexpected error occurred during imports from set_env: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None
    embeddings_model = None
    qdrant_collection_name = "unknown_collection"

# Define the @tool for retrieval
@tool
def retrieve_documents(query: str) -> dict:
    """
    Retrieve relevant documents from the Qdrant vector store based on the user's query.
    The query should be a self-contained question, potentially rephrased from earlier conversation.
    """
    print(f"--- main.py (v2) retrieve_documents TOOL CALLED with query: '{query}' for collection '{qdrant_collection_name}'"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (v2) retrieve_documents ERROR: vector_store is not available."); sys.stdout.flush()
        # Return a structure that ToolNode can handle and can be identified as an error
        return {"error": "Vector store not available.", "documents_content": "Vector store is offline.", "retrieved_docs_metadata": []}
    try:
        # Using k=3 for a balance. Adjust as needed.
        retrieved_docs = vector_store.similarity_search(query, k=3) 
        print(f"--- main.py (v2) retrieve_documents: Retrieved {len(retrieved_docs)} documents."); sys.stdout.flush()
        
        if not retrieved_docs:
            return {"error": None, "documents_content": "No specific documents found for this query in the knowledge base.", "retrieved_docs_metadata": []}

        # Serialize documents for the ToolMessage content and keep metadata
        docs_content_for_llm = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))}\nContent: {doc.page_content}" 
            for doc in retrieved_docs
        ])
        retrieved_docs_metadata = [doc.metadata for doc in retrieved_docs] # Potentially useful for frontend display later

        # The content of the ToolMessage should ideally be a string if it's directly used by LLM.
        # Here, we are passing a dictionary, which is fine as ToolNode will handle it.
        # The generate_answer_node will then parse this dictionary.
        return {
            "error": None, 
            "documents_content": docs_content_for_llm, 
            "retrieved_docs_metadata": retrieved_docs_metadata 
        }
    except Exception as e:
        print(f"--- main.py (v2) retrieve_documents ERROR during similarity search: {e}", exc_info=True); sys.stdout.flush()
        return {"error": f"Retrieval failed: {e}", "documents_content": "Error during document retrieval.", "retrieved_docs_metadata": []}

# Nodes for the graph

def query_router_node(state: MessagesState):
    """
    This node decides whether to directly answer the user or to call the retrieval tool.
    It uses the LLM with tool-binding to make this decision.
    The input 'state' is a list of messages (history + current user query).
    """
    print("--- main.py (v2) Query Router Node ---"); sys.stdout.flush()
    print(f"--- main.py (v2) Query Router Node: Received messages: {state['messages']}"); sys.stdout.flush()
    if llm is None:
        print("--- main.py (v2) Query Router Node: LLM is not available."); sys.stdout.flush()
        # Append an error message to the state and let it flow, or return immediately
        return {"messages": [AIMessage(content="LLM is not available. Cannot process request.")]}

    llm_with_tools = llm.bind_tools([retrieve_documents])
    
    try:
        ai_response_or_tool_call = llm_with_tools.invoke(state["messages"])
        print(f"--- main.py (v2) Query Router Node: LLM response/tool_call: {ai_response_or_tool_call}"); sys.stdout.flush()
        return {"messages": [ai_response_or_tool_call]}
    except Exception as e:
        print(f"--- main.py (v2) Query Router Node: Error invoking LLM with tools: {e}", exc_info=True); sys.stdout.flush()
        return {"messages": [AIMessage(content=f"Error in query router: {e}")]}


tool_node = ToolNode([retrieve_documents]) 

def generate_answer_node(state: MessagesState):
    """
    Generates the final answer using the LLM, considering the original user query,
    chat history, and any retrieved documents (from ToolMessage).
    """
    print("--- main.py (v2) Generate Answer Node ---"); sys.stdout.flush()
    print(f"--- main.py (v2) Generate Answer Node: Current messages state: {state['messages']}"); sys.stdout.flush()

    if llm is None:
        print("--- main.py (v2) Generate Answer Node: LLM is not available."); sys.stdout.flush()
        return {"messages": [AIMessage(content="LLM is not available. Cannot generate answer.")]}

    current_messages = state["messages"]
    
    retrieved_context_str = "No documents were retrieved or an error occurred during retrieval."
    # Extract retrieved documents if a ToolMessage is present from our retrieve_documents tool
    # The ToolNode adds the tool output as the content of the ToolMessage.
    # The content of the ToolMessage is the dictionary returned by retrieve_documents.
    for msg in reversed(current_messages): # Look for the most recent ToolMessage
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_documents":
            tool_output = msg.content # This should be the dictionary
            if isinstance(tool_output, dict):
                if tool_output.get("error"):
                    retrieved_context_str = f"Note on retrieval: {tool_output.get('error')}. Document content: {tool_output.get('documents_content', '')}"
                else:
                    retrieved_context_str = tool_output.get("documents_content", "Retrieved content not found in tool output.")
            else: # Should not happen if tool returns dict
                retrieved_context_str = f"Unexpected tool output format: {tool_output}"
            break 
    
    # Find the most recent HumanMessage as "the question" for the final prompt
    last_human_message_content = "User question not clearly identified in recent history."
    for msg in reversed(current_messages):
        if isinstance(msg, HumanMessage):
            last_human_message_content = msg.content
            break
    
    # Build a simplified history string for the prompt (excluding tool calls and tool messages for brevity)
    history_str_parts = []
    for msg in current_messages:
        if isinstance(msg, HumanMessage):
            history_str_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls: # Only AI direct responses
            history_str_parts.append(f"Assistant: {msg.content}")
    formatted_chat_history = "\n".join(history_str_parts)
    if not formatted_chat_history:
        formatted_chat_history = "No prior conversation."

    # Construct the prompt for the LLM
    # This prompt template is taken from the Langchain tutorial for conversational RAG
    prompt_template_str = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Retrieved Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    final_prompt = ChatPromptTemplate.from_template(prompt_template_str)
    
    chain = final_prompt | llm
    
    print(f"--- main.py (v2) Generate Answer Node: Generating final answer for question: '{last_human_message_content}' with context snippet: '{retrieved_context_str[:200]}...'"); sys.stdout.flush()

    try:
        ai_final_answer = chain.invoke({
            "question": last_human_message_content, 
            "context": retrieved_context_str,
            "chat_history": formatted_chat_history
        })
        print(f"--- main.py (v2) Generate Answer Node: Final AI Answer: {ai_final_answer.content}"); sys.stdout.flush()
        return {"messages": [ai_final_answer]} # Append the final AI answer
    except Exception as e:
        print(f"--- main.py (v2) Generate Answer Node: Error invoking LLM for final answer: {e}", exc_info=True); sys.stdout.flush()
        return {"messages": [AIMessage(content=f"Error generating final answer: {e}")]}


# Compile the graph
graph = None
if llm and vector_store and embeddings_model: # Ensure all key components are loaded
    print("--- main.py (v2 - Conversational RAG with mpnet): Compiling LangGraph ---"); sys.stdout.flush()
    workflow = StateGraph(MessagesState) 

    workflow.add_node("query_router", query_router_node)
    workflow.add_node("retriever_tool_node", tool_node) 
    workflow.add_node("answer_generator", generate_answer_node)

    workflow.set_entry_point("query_router")
    
    # If query_router calls a tool, messages are updated with AIMessage(tool_calls=...).
    # tools_condition checks for 'tool_calls'. If present, it routes to the key provided for tools.
    # If not present (LLM answers directly), it routes to END.
    # In our case, if no tool is called, the AIMessage from query_router IS the final answer.
    workflow.add_conditional_edges(
        "query_router",
        tools_condition, 
        {"tools": "retriever_tool_node", END: END}, 
    )
    
    workflow.add_edge("retriever_tool_node", "answer_generator")
    workflow.add_edge("answer_generator", END) 

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    print("--- main.py (v2 - Conversational RAG with mpnet): Conversational RAG graph compiled successfully with MemorySaver."); sys.stdout.flush()
else:
    if not llm: print("--- main.py (v2) ERROR: LLM is None, cannot compile graph."); sys.stdout.flush()
    if not vector_store: print("--- main.py (v2) ERROR: vector_store is None, cannot compile graph."); sys.stdout.flush()
    if not embeddings_model: print("--- main.py (v2) ERROR: embeddings_model is None, cannot compile graph."); sys.stdout.flush()
    graph = None

