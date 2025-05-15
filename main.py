# main.py (v2 - Conversational RAG with Tool Calling)
import sys
print("--- main.py (v2) IMPORTING ---"); sys.stdout.flush()

from typing import List, Optional, Annotated, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState # Use MessagesState
from langgraph.prebuilt import ToolNode, tools_condition # For conditional tool execution
from langgraph.checkpoint.memory import MemorySaver # For in-memory chat history

# Import llm and vector_store from set_env
# These should be initialized by the time this module is imported by app.py
try:
    from set_env import llm, vector_store, collection_name as qdrant_collection_name # Also get collection_name
    if llm is None:
        print("--- main.py (v2) ERROR: LLM not initialized in set_env.py"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (v2) ERROR: vector_store not initialized in set_env.py. RAG will not function."); sys.stdout.flush()
    else:
        print(f"--- main.py (v2): Successfully imported llm and vector_store for collection '{qdrant_collection_name}'."); sys.stdout.flush()

except ImportError as e:
    print(f"--- main.py (v2) ERROR: Could not import from set_env.py: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None
    qdrant_collection_name = "unknown_collection"
except Exception as e:
    print(f"--- main.py (v2) An unexpected error occurred during imports from set_env: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None
    qdrant_collection_name = "unknown_collection"

# Define the @tool for retrieval
@tool
def retrieve_documents(query: str) -> dict:
    """
    Retrieve relevant documents from the Qdrant vector store based on the user's query.
    The query should be a self-contained question, potentially rephrased from earlier conversation.
    """
    print(f"--- main.py (v2) retrieve_documents TOOL CALLED with query: '{query}'"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (v2) retrieve_documents ERROR: vector_store is not available."); sys.stdout.flush()
        return {"error": "Vector store not available.", "documents_content": "", "retrieved_docs_metadata": []}
    try:
        retrieved_docs = vector_store.similarity_search(query, k=3) # Retrieve top 3
        print(f"--- main.py (v2) retrieve_documents: Retrieved {len(retrieved_docs)} documents."); sys.stdout.flush()
        
        if not retrieved_docs:
            return {"error": None, "documents_content": "No specific documents found for this query.", "retrieved_docs_metadata": []}

        # Serialize documents for the ToolMessage content and keep metadata
        docs_content_for_llm = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))}\nContent: {doc.page_content}" 
            for doc in retrieved_docs
        ])
        retrieved_docs_metadata = [doc.metadata for doc in retrieved_docs]

        return {
            "error": None, 
            "documents_content": docs_content_for_llm, 
            "retrieved_docs_metadata": retrieved_docs_metadata
        }
    except Exception as e:
        print(f"--- main.py (v2) retrieve_documents ERROR during similarity search: {e}"); sys.stdout.flush()
        return {"error": f"Retrieval failed: {e}", "documents_content": "", "retrieved_docs_metadata": []}

# Nodes for the graph

def query_router_node(state: MessagesState):
    """
    This node decides whether to directly answer the user or to call the retrieval tool.
    It uses the LLM with tool-binding to make this decision.
    The input 'state' is a list of messages (history + current user query).
    """
    print("--- main.py (v2) Query Router Node ---"); sys.stdout.flush()
    if llm is None:
        return {"messages": [AIMessage(content="LLM is not available. Cannot process request.")]}

    # Bind the retrieval tool to the LLM
    # Only provide tools that the LLM can choose to call.
    llm_with_tools = llm.bind_tools([retrieve_documents])
    
    # Invoke the LLM with the current message history.
    # MessagesState appends, so the last message is the current user query.
    ai_response_or_tool_call = llm_with_tools.invoke(state["messages"])
    
    # The response will either be an AIMessage with content (if LLM answers directly)
    # or an AIMessage with 'tool_calls' if it decides to use the retrieve_documents tool.
    return {"messages": [ai_response_or_tool_call]}


tool_node = ToolNode([retrieve_documents]) # Handles execution of the retrieve_documents tool

def generate_answer_node(state: MessagesState):
    """
    Generates the final answer using the LLM, considering the original user query,
    chat history, and any retrieved documents (from ToolMessage).
    """
    print("--- main.py (v2) Generate Answer Node ---"); sys.stdout.flush()
    if llm is None:
        return {"messages": [AIMessage(content="LLM is not available. Cannot generate answer.")]}

    # The last message in 'state["messages"]' could be an AIMessage from query_router (if no tool call)
    # or a ToolMessage from tool_node (if retrieval happened).
    
    current_messages = state["messages"]
    
    # Extract user questions and AI responses for context, and tool messages for retrieved docs
    conversation_history_for_prompt = []
    retrieved_context_str = "No documents were retrieved or retrieval was not attempted for this query."
    
    # The last message is the one we're responding to, or the tool result.
    # The actual user question that triggered this flow is typically the second to last human message if a tool was called,
    # or the last human message if no tool was called.
    # For simplicity, let's find the most recent HumanMessage as "the question".
    
    last_human_message_content = "User question not clearly identified in history."
    for msg in reversed(current_messages):
        if isinstance(msg, HumanMessage):
            last_human_message_content = msg.content
            break

    # Extract retrieved documents if a ToolMessage is present from our retrieve_documents tool
    for msg in reversed(current_messages):
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_documents":
            tool_output = msg.content # msg.content here is the dict we returned from retrieve_documents
            if isinstance(tool_output, str): # If tool returned a simple string by mistake
                retrieved_context_str = tool_output
            elif isinstance(tool_output, dict):
                 retrieved_context_str = tool_output.get("documents_content", "Error parsing tool output.")
            break 
            # Assuming only one retrieval result to consider for this simple RAG

    # Build a simplified history string for the prompt
    history_str_parts = []
    for msg in current_messages:
        if not isinstance(msg, ToolMessage): # Exclude bulky tool messages from chat history string for prompt
            if isinstance(msg, HumanMessage):
                history_str_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content: # Only include AI messages with actual content
                history_str_parts.append(f"Assistant: {msg.content}")
    formatted_chat_history = "\n".join(history_str_parts)


    # Construct the prompt for the LLM
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=f"You are a helpful assistant for question-answering tasks based on the provided document context. "
                    f"Use the following pieces of retrieved context to answer the user's question. "
                    f"If the context doesn't provide an answer, say that you don't know or cannot answer based on the provided document. "
                    f"Keep the answer concise and relevant to the question. Refer to the chat history for conversational context.\n\n"
                    f"Chat History:\n{formatted_chat_history}\n\n" # Add formatted history here
                    f"Retrieved Context from Document:\n{retrieved_context_str}"
        ),
        HumanMessage(content="{question}") # Placeholder for the user's most recent question
    ])
    
    chain = prompt_template | llm
    
    ai_final_answer = chain.invoke({"question": last_human_message_content})
    
    print(f"--- main.py (v2) Generate Answer Node: Final AI Answer: {ai_final_answer.content}"); sys.stdout.flush()
    return {"messages": [ai_final_answer]}


# Compile the graph
graph = None
if llm and vector_store:
    print("--- main.py (v2) Compiling LangGraph ---"); sys.stdout.flush()
    workflow = StateGraph(MessagesState) # Use MessagesState for history

    workflow.add_node("query_router", query_router_node)
    workflow.add_node("retriever_tool_node", tool_node) # Using the prebuilt ToolNode
    workflow.add_node("answer_generator", generate_answer_node)

    workflow.set_entry_point("query_router")

    # Conditional edge: if the LLM in query_router decides to call a tool, route to retriever_tool_node.
    # Otherwise (LLM answers directly), route to END (or to answer_generator if direct LLM response needs final formatting).
    # tools_condition will route to "tools" (our retriever_tool_node name) or END.
    # We want to always go to answer_generator after any tool call or direct LLM response from query_router
    # For simplicity here, if query_router calls a tool, it goes to tool_node, then to answer_generator.
    # If query_router generates a direct AI message (no tool call), that IS the answer.
    # The tutorial's tools_condition routes to END if no tool is called. Let's adapt.
    
    # If query_router calls a tool, messages are updated with AIMessage(tool_calls=...).
    # tools_condition checks for 'tool_calls'. If present, it routes to the key provided for tools.
    # If not present, it routes to END.
    workflow.add_conditional_edges(
        "query_router",
        tools_condition, # LangGraph's prebuilt condition
        # If tools_condition returns "tools", then call 'retriever_tool_node'.
        # If tools_condition returns END (meaning no tool calls), then the AIMessage from query_router is the final one.
        {"tools": "retriever_tool_node", END: END}, 
    )
    
    # After tools are called (retriever_tool_node), the result (ToolMessage) is added to state.
    # Then, go to the answer_generator.
    workflow.add_edge("retriever_tool_node", "answer_generator")
    workflow.add_edge("answer_generator", END) # Final answer generated, go to END.

    # Initialize memory (checkpointer)
    # For a deployed app, you'd use a persistent checkpointer (e.g., SqliteSaver, RedisSaver, or a custom one)
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    print("--- main.py (v2) Conversational RAG graph compiled successfully with MemorySaver."); sys.stdout.flush()
else:
    print("--- main.py (v2) RAG graph could not be compiled due to missing LLM or vector_store."); sys.stdout.flush()

# Example of how app.py might call it:
# config = {"configurable": {"thread_id": "some_unique_thread_id"}}
# new_user_message = HumanMessage(content="What is task decomposition?")
# for event in graph.stream({"messages": [new_user_message]}, config=config, stream_mode="values"):
#     last_message = event["messages"][-1]
#     # last_message will be AIMessage (direct answer), AIMessage (with tool call), ToolMessage, or final AIMessage
#     print(f"Event: {last_message.type} - Content: {last_message.content if isinstance(last_message, (HumanMessage, AIMessage, SystemMessage)) else 'Tool data'}")
#     if isinstance(last_message, AIMessage) and not last_message.tool_calls and last_message.content:
#         final_answer_for_client = last_message.content 
#         # (app.py would send this back)
