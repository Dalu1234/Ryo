# main.py (Conversational FINAL - v1.2 - Fix Query Router Tool Usage)
import sys
print("--- main.py (CONVERSATIONAL FINAL v1.2) IMPORTING ---"); sys.stdout.flush() # Updated version
import logging 

from typing import List, Optional, Annotated, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --- Attempt to import shared components ---
try:
    from set_env import llm, vector_store, embeddings_model, collection_name as qdrant_collection_name
    if llm is None:
        print("--- main.py (CONVERSATIONAL FINAL v1.2) ERROR: LLM not initialized in set_env.py"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (CONVERSATIONAL FINAL v1.2) ERROR: vector_store not initialized in set_env.py. RAG will not function."); sys.stdout.flush()
    if embeddings_model is None:
        print("--- main.py (CONVERSATIONAL FINAL v1.2) ERROR: embeddings_model not initialized in set_env.py."); sys.stdout.flush()
    else:
        model_name_display = getattr(embeddings_model, 'model_name', 'Unknown Model')
        print(f"--- main.py (CONVERSATIONAL FINAL v1.2): Successfully imported llm, vector_store (for collection '{qdrant_collection_name}'), and embeddings_model '{model_name_display}'."); sys.stdout.flush()

except ImportError as e:
    logging.error(f"--- main.py (CONVERSATIONAL FINAL v1.2) ERROR: Could not import from set_env.py: {e}", exc_info=True); sys.stdout.flush()
    llm = None
    vector_store = None
    embeddings_model = None
    qdrant_collection_name = "allocations_docs" # Fallback
except Exception as e:
    logging.error(f"--- main.py (CONVERSATIONAL FINAL v1.2) An unexpected error during imports from set_env: {e}", exc_info=True); sys.stdout.flush()
    llm = None
    vector_store = None
    embeddings_model = None
    qdrant_collection_name = "allocations_docs" # Fallback

# --- Tool Definition ---
@tool
def retrieve_documents(query: str) -> dict:
    """
    Retrieve relevant documents from the Qdrant vector store based on the user's query.
    The query should be a self-contained question about the 'Allocations Manual', potentially rephrased from earlier conversation.
    Only use this tool if the user's question explicitly asks for information that can be found in the Allocations Manual.
    """
    print(f"--- main.py (CONVERSATIONAL FINAL v1.2) retrieve_documents TOOL CALLED with query: '{query}' for collection '{qdrant_collection_name}'"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (CONVERSATIONAL FINAL v1.2) retrieve_documents ERROR: vector_store is not available."); sys.stdout.flush()
        return {"error": "Vector store not available.", "documents_content": "Vector store is offline.", "retrieved_docs_metadata": []}
    try:
        retrieved_docs = vector_store.similarity_search(query, k=3)
        print(f"--- main.py (CONVERSATIONAL FINAL v1.2) retrieve_documents: Retrieved {len(retrieved_docs)} documents."); sys.stdout.flush()

        if not retrieved_docs:
            return {"error": None, "documents_content": "No specific documents found for this query in the Allocations Manual.", "retrieved_docs_metadata": []}

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
        logging.error(f"--- main.py (CONVERSATIONAL FINAL v1.2) retrieve_documents ERROR during similarity search: {e}", exc_info=True); sys.stdout.flush()
        return {"error": f"Retrieval failed: {e}", "documents_content": "Error during document retrieval.", "retrieved_docs_metadata": []}

# --- Graph Nodes ---
def query_router_node(state: MessagesState):
    print("--- main.py (CONVERSATIONAL FINAL v1.2) Query Router Node ---"); sys.stdout.flush()
    log_messages_summary = [f"{msg.type}: {msg.content[:50] + '...' if len(msg.content) > 50 else msg.content}" for msg in state['messages']]
    print(f"--- main.py (CONVERSATIONAL FINAL v1.2) Query Router Node: Received messages summary: {log_messages_summary}"); sys.stdout.flush()

    if llm is None:
        print("--- main.py (CONVERSATIONAL FINAL v1.2) Query Router Node: LLM is not available."); sys.stdout.flush()
        return {"messages": [AIMessage(content="LLM is not available. Cannot process request.")]}

    # System message to guide the LLM's routing decision and tool use
    system_message_content = (
        "You are an AI assistant acting as a query router for a RAG (Retrieval Augmented Generation) system. "
        "Your purpose is to analyze the user's latest question and decide the best course of action. "
        "You have access to ONLY ONE tool: 'retrieve_documents'. This tool should be used to fetch relevant information from the 'Allocations Manual' if the user's question requires it. "
        "Follow these rules strictly:\n"
        "1. If the user's question can be answered with information likely found in the 'Allocations Manual', you MUST use the 'retrieve_documents' tool. Rephrase the user's question into a self-contained query for the tool if necessary.\n"
        "2. If the user's question is a simple greeting, a conversational follow-up that doesn't require new information from the manual, or clearly out of scope (e.g., asking for the weather, personal opinions, or to perform actions like sending emails), then respond directly without using any tools.\n"
        "3. Do NOT attempt to call or invent any other tools or functions besides 'retrieve_documents'. If you think another tool is needed, state that you cannot perform that action and can only retrieve documents related to the Allocations Manual."
    )
    
    messages_for_llm = [SystemMessage(content=system_message_content)]
    # Add existing messages, filtering out any identical system messages to avoid duplication if this node were part of a loop.
    for msg in state["messages"]:
        if not (isinstance(msg, SystemMessage) and msg.content == system_message_content):
             messages_for_llm.append(msg)

    llm_with_tools = llm.bind_tools([retrieve_documents])

    try:
        ai_response_or_tool_call = llm_with_tools.invoke(messages_for_llm)
        
        tool_call_summary = "None"
        if ai_response_or_tool_call.tool_calls:
            tool_call_summary = str([tc['name'] for tc in ai_response_or_tool_call.tool_calls])
        
        print(f"--- main.py (CONVERSATIONAL FINAL v1.2) Query Router Node: LLM response/tool_call: {ai_response_or_tool_call.type} - Content: {ai_response_or_tool_call.content[:100] if ai_response_or_tool_call.content else 'N/A'} - ToolCalls: {tool_call_summary}"); sys.stdout.flush()
        
        # Additional check: Ensure only the allowed tool is called
        if ai_response_or_tool_call.tool_calls:
            for tc in ai_response_or_tool_call.tool_calls:
                if tc['name'] != 'retrieve_documents':
                    error_message = f"I apologize, I tried to use a function called '{tc['name']}' which isn't available to me. I can only retrieve documents from the Allocations Manual or chat directly."
                    print(f"--- main.py (CONVERSATIONAL FINAL v1.2) Query Router Node WARNING: LLM attempted to call an unauthorized tool: {tc['name']}. Overriding to respond: {error_message}"); sys.stdout.flush()
                    return {"messages": [AIMessage(content=error_message)]} # Return an AIMessage instead of the invalid tool call
        
        return {"messages": [ai_response_or_tool_call]}
    except Exception as e:
        logging.error(f"--- main.py (CONVERSATIONAL FINAL v1.2) Query Router Node: Error invoking LLM with tools: {e}", exc_info=True); sys.stdout.flush()
        # Include the specific error message from the LLM if available in the exception
        error_detail = str(e)
        return {"messages": [AIMessage(content=f"Error in query router: {error_detail}")]}

tool_node = ToolNode([retrieve_documents])

def generate_answer_node(state: MessagesState):
    print("--- main.py (CONVERSATIONAL FINAL v1.2) Generate Answer Node ---"); sys.stdout.flush()
    if llm is None:
        print("--- main.py (CONVERSATIONAL FINAL v1.2) Generate Answer Node: LLM is not available."); sys.stdout.flush()
        return {"messages": [AIMessage(content="LLM is not available. Cannot generate answer.")]}

    current_messages = state["messages"]
    retrieved_context_str = "No documents were retrieved or an error occurred during retrieval."
    # Iterate to find the latest ToolMessage which contains the output of retrieve_documents
    for msg in reversed(current_messages):
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_documents":
            # The content of ToolMessage is expected to be a string (often JSON string) 
            # or a dict if the tool returns a dict directly and LangChain handles it.
            # Based on your retrieve_documents tool, it returns a dict.
            tool_output = msg.content 
            
            # Check if tool_output is already a dict (LangChain might parse it if it's JSON)
            # Or if it's a string that needs to be parsed (less likely if tool returns dict)
            # For now, assuming msg.content is the dict returned by your tool
            if isinstance(tool_output, dict):
                if tool_output.get("error"):
                    retrieved_context_str = f"Note on retrieval: {tool_output.get('error')}. Document content: {tool_output.get('documents_content', '')}"
                else:
                    retrieved_context_str = tool_output.get("documents_content", "Retrieved content not found in tool output.")
            else: # If tool_output is not a dict (e.g., it's a string representation)
                retrieved_context_str = f"Tool output was not in the expected dictionary format. Received: {str(tool_output)[:200]}..."
            break 
            # If your tool was returning a JSON string, you'd need:
            # import json
            # try:
            #   tool_output_dict = json.loads(msg.content)
            #   if tool_output_dict.get("error"): ... else: ...
            # except json.JSONDecodeError:
            #   retrieved_context_str = "Error decoding tool output."

    last_human_message_content = "User question not clearly identified in recent history."
    full_chat_history_for_prompt = []

    for msg in current_messages:
        if isinstance(msg, HumanMessage):
            last_human_message_content = msg.content
            full_chat_history_for_prompt.append(msg)
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls: # Only include AI messages that are direct responses
            full_chat_history_for_prompt.append(msg)

    # Create a concise history string, excluding the very last human message (which is the current question)
    formatted_chat_history_for_llm = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}" for m in full_chat_history_for_prompt[:-1]]
    )
    if not formatted_chat_history_for_llm:
         formatted_chat_history_for_llm = "No prior turns in this conversation."

    prompt_template_str = (
        "You are a helpful AI assistant for question-answering tasks about the 'Allocations Manual'. "
        "Use ONLY the following pieces of 'Retrieved Context from Document' to answer the 'Current User Question'. "
        "If the retrieved context does not contain the answer, or if it indicates no relevant documents were found, state that you cannot answer based on the provided information from the Allocations Manual. "
        "Do not use any external knowledge or make assumptions beyond the provided context. "
        "Keep your answer concise and directly related to the question and context. Maximum three sentences."
        "\n\n"
        "Chat History (earlier turns, for context only, do not use to answer current question):\n{chat_history}\n\n"
        "Retrieved Context from Document:\n{context}\n\n"
        "Current User Question: {question}\n\n"
        "Answer (based ONLY on the Retrieved Context from Document):"
    )
    final_prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = final_prompt | llm

    print(f"--- main.py (CONVERSATIONAL FINAL v1.2) Generate Answer Node: Generating final answer for question: '{last_human_message_content}' with context snippet: '{retrieved_context_str[:200]}...' and history snippet: '{formatted_chat_history_for_llm[:100]}...'"); sys.stdout.flush()

    try:
        ai_final_answer = chain.invoke({
            "question": last_human_message_content,
            "context": retrieved_context_str,
            "chat_history": formatted_chat_history_for_llm
        })
        print(f"--- main.py (CONVERSATIONAL FINAL v1.2) Generate Answer Node: Final AI Answer: {ai_final_answer.content}"); sys.stdout.flush()
        return {"messages": [ai_final_answer]}
    except Exception as e:
        logging.error(f"--- main.py (CONVERSATIONAL FINAL v1.2) Generate Answer Node: Error invoking LLM for final answer: {e}", exc_info=True); sys.stdout.flush()
        return {"messages": [AIMessage(content=f"Error generating final answer: {e}")]}

# --- Graph Compilation ---
graph = None
if llm and vector_store and embeddings_model:
    print("--- main.py (CONVERSATIONAL FINAL v1.2): Compiling LangGraph ---"); sys.stdout.flush()
    workflow = StateGraph(MessagesState)

    workflow.add_node("query_router", query_router_node)
    workflow.add_node("retriever_tool_node", tool_node) # This node executes the tool(s)
    workflow.add_node("answer_generator", generate_answer_node)

    workflow.set_entry_point("query_router")

    # Conditional edge from query_router:
    # If LLM in query_router decides to call a tool, flow to retriever_tool_node.
    # Otherwise (LLM responds directly), flow to END.
    workflow.add_conditional_edges(
        "query_router",
        tools_condition, # This prebuilt condition checks for tool_calls in the AIMessage
        {"tools": "retriever_tool_node", END: END}, # 'tools' is the key if tool_calls exist
    )

    # After tool execution, always go to the answer_generator
    workflow.add_edge("retriever_tool_node", "answer_generator")
    # After generating an answer, end the flow
    workflow.add_edge("answer_generator", END)

    memory = MemorySaver() # In-memory checkpointer for storing conversation state
    graph = workflow.compile(checkpointer=memory)
    print("--- main.py (CONVERSATIONAL FINAL v1.2): Conversational RAG graph compiled successfully with MemorySaver."); sys.stdout.flush()
else:
    # Log which component is missing
    if not llm: print("--- main.py (CONVERSATIONAL FINAL v1.2) ERROR: LLM is None, cannot compile graph."); sys.stdout.flush()
    if not vector_store: print("--- main.py (CONVERSATIONAL FINAL v1.2) ERROR: vector_store is None, cannot compile graph."); sys.stdout.flush()
    if not embeddings_model: print("--- main.py (CONVERSATIONAL FINAL v1.2) ERROR: embeddings_model is None, cannot compile graph."); sys.stdout.flush()
    graph = None
