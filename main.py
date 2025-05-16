# main.py (Conversational FINAL - for all-mpnet-base-v2 model)
import sys
import logging
print("--- main.py (CONVERSATIONAL FINAL) IMPORTING ---"); sys.stdout.flush()

from typing import List, Optional, Annotated, TypedDict
from langchain_core.documents import Document 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage 
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate 

from langgraph.graph import StateGraph, END, MessagesState 
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.checkpoint.memory import MemorySaver 

try:
    from set_env import llm, vector_store, embeddings_model, collection_name as qdrant_collection_name
    if llm is None:
        print("--- main.py (CONVERSATIONAL FINAL) ERROR: LLM not initialized in set_env.py"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (CONVERSATIONAL FINAL) ERROR: vector_store not initialized in set_env.py. RAG will not function."); sys.stdout.flush()
    if embeddings_model is None: 
        print("--- main.py (CONVERSATIONAL FINAL) ERROR: embeddings_model not initialized in set_env.py."); sys.stdout.flush()
    else:
        print(f"--- main.py (CONVERSATIONAL FINAL): Successfully imported llm, vector_store (for collection '{qdrant_collection_name}'), and embeddings_model '{embeddings_model.model_name}'."); sys.stdout.flush()

except ImportError as e:
    print(f"--- main.py (CONVERSATIONAL FINAL) ERROR: Could not import from set_env.py: {e}"); sys.stdout.flush()
    llm = None
    vector_store = None
    embeddings_model = None
    qdrant_collection_name = "allocations_docs" # Fallback, should match set_env.py
except Exception as e:
    logging.error(f"An unexpected error during imports from set_env: {e}", exc_info=True)
    llm = None
    vector_store = None
    embeddings_model = None
    qdrant_collection_name = "allocations_docs" # Fallback

@tool
def retrieve_documents(query: str) -> dict:
    """
    Retrieve relevant documents from the Qdrant vector store based on the user's query.
    The query should be a self-contained question, potentially rephrased from earlier conversation.
    """
    print(f"--- main.py (CONVERSATIONAL FINAL) retrieve_documents TOOL CALLED with query: '{query}' for collection '{qdrant_collection_name}'"); sys.stdout.flush()
    if vector_store is None:
        print("--- main.py (CONVERSATIONAL FINAL) retrieve_documents ERROR: vector_store is not available."); sys.stdout.flush()
        return {"error": "Vector store not available.", "documents_content": "Vector store is offline.", "retrieved_docs_metadata": []}
    try:
        retrieved_docs = vector_store.similarity_search(query, k=3) 
        print(f"--- main.py (CONVERSATIONAL FINAL) retrieve_documents: Retrieved {len(retrieved_docs)} documents."); sys.stdout.flush()
        
        if not retrieved_docs:
            return {"error": None, "documents_content": "No specific documents found for this query in the knowledge base.", "retrieved_docs_metadata": []}

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
        print(f"--- main.py (CONVERSATIONAL FINAL) retrieve_documents ERROR during similarity search: {e}", exc_info=True); sys.stdout.flush()
        return {"error": f"Retrieval failed: {e}", "documents_content": "Error during document retrieval.", "retrieved_docs_metadata": []}

def query_router_node(state: MessagesState):
    print("--- main.py (CONVERSATIONAL FINAL) Query Router Node ---"); sys.stdout.flush()
    # Log a summary of messages to avoid overly verbose logs if history is long
    log_messages_summary = [f"{msg.type}: {msg.content[:50] + '...' if len(msg.content) > 50 else msg.content}" for msg in state['messages']]
    print(f"--- main.py (CONVERSATIONAL FINAL) Query Router Node: Received messages summary: {log_messages_summary}"); sys.stdout.flush()

    if llm is None:
        print("--- main.py (CONVERSATIONAL FINAL) Query Router Node: LLM is not available."); sys.stdout.flush()
        return {"messages": [AIMessage(content="LLM is not available. Cannot process request.")]}

    llm_with_tools = llm.bind_tools([retrieve_documents])
    
    try:
        ai_response_or_tool_call = llm_with_tools.invoke(state["messages"])
        tool_call_summary = "None"
        if ai_response_or_tool_call.tool_calls:
            tool_call_summary = str([tc['name'] for tc in ai_response_or_tool_call.tool_calls])
        print(f"--- main.py (CONVERSATIONAL FINAL) Query Router Node: LLM response/tool_call: {ai_response_or_tool_call.type} - Content: {ai_response_or_tool_call.content[:100] if ai_response_or_tool_call.content else 'N/A'} - ToolCalls: {tool_call_summary}"); sys.stdout.flush()
        return {"messages": [ai_response_or_tool_call]}
    except Exception as e:
        print(f"--- main.py (CONVERSATIONAL FINAL) Query Router Node: Error invoking LLM with tools: {e}", exc_info=True); sys.stdout.flush()
        return {"messages": [AIMessage(content=f"Error in query router: {e}")]}

tool_node = ToolNode([retrieve_documents]) 

def generate_answer_node(state: MessagesState):
    print("--- main.py (CONVERSATIONAL FINAL) Generate Answer Node ---"); sys.stdout.flush()
    if llm is None:
        print("--- main.py (CONVERSATIONAL FINAL) Generate Answer Node: LLM is not available."); sys.stdout.flush()
        return {"messages": [AIMessage(content="LLM is not available. Cannot generate answer.")]}

    current_messages = state["messages"]
    retrieved_context_str = "No documents were retrieved or an error occurred during retrieval."
    for msg in reversed(current_messages): 
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_documents":
            tool_output = msg.content 
            if isinstance(tool_output, dict):
                if tool_output.get("error"):
                    retrieved_context_str = f"Note on retrieval: {tool_output.get('error')}. Document content: {tool_output.get('documents_content', '')}"
                else:
                    retrieved_context_str = tool_output.get("documents_content", "Retrieved content not found in tool output.")
            else: 
                retrieved_context_str = f"Unexpected tool output format: {tool_output}"
            break 
    
    last_human_message_content = "User question not clearly identified in recent history."
    full_chat_history_for_prompt = [] 

    for msg in current_messages:
        if isinstance(msg, HumanMessage):
            last_human_message_content = msg.content 
            full_chat_history_for_prompt.append(msg)
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls: 
            full_chat_history_for_prompt.append(msg)

    # Format chat history for the prompt (excluding the very last human message which is the current question)
    formatted_chat_history_for_llm = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}" for m in full_chat_history_for_prompt[:-1]] 
    )
    if not formatted_chat_history_for_llm:
         formatted_chat_history_for_llm = "No prior turns in this conversation."

    prompt_template_str = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "Chat History (earlier turns):\n{chat_history}\n\n"
        "Retrieved Context from Document:\n{context}\n\n"
        "Current User Question: {question}\n\n"
        "Answer:"
    )
    final_prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = final_prompt | llm
    
    print(f"--- main.py (CONVERSATIONAL FINAL) Generate Answer Node: Generating final answer for question: '{last_human_message_content}' with context snippet: '{retrieved_context_str[:200]}...' and history snippet: '{formatted_chat_history_for_llm[:200]}...'"); sys.stdout.flush()

    try:
        ai_final_answer = chain.invoke({
            "question": last_human_message_content, 
            "context": retrieved_context_str,
            "chat_history": formatted_chat_history_for_llm
        })
        print(f"--- main.py (CONVERSATIONAL FINAL) Generate Answer Node: Final AI Answer: {ai_final_answer.content}"); sys.stdout.flush()
        return {"messages": [ai_final_answer]} 
    except Exception as e:
        print(f"--- main.py (CONVERSATIONAL FINAL) Generate Answer Node: Error invoking LLM for final answer: {e}", exc_info=True); sys.stdout.flush()
        return {"messages": [AIMessage(content=f"Error generating final answer: {e}")]}

graph = None
if llm and vector_store and embeddings_model: 
    print("--- main.py (CONVERSATIONAL FINAL): Compiling LangGraph ---"); sys.stdout.flush()
    workflow = StateGraph(MessagesState) 

    workflow.add_node("query_router", query_router_node)
    workflow.add_node("retriever_tool_node", tool_node) 
    workflow.add_node("answer_generator", generate_answer_node)

    workflow.set_entry_point("query_router")
    
    workflow.add_conditional_edges(
        "query_router",
        tools_condition, 
        {"tools": "retriever_tool_node", END: END}, 
    )
    
    workflow.add_edge("retriever_tool_node", "answer_generator")
    workflow.add_edge("answer_generator", END) 

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    print("--- main.py (CONVERSATIONAL FINAL): Conversational RAG graph compiled successfully with MemorySaver."); sys.stdout.flush()
else:
    if not llm: print("--- main.py (CONVERSATIONAL FINAL) ERROR: LLM is None, cannot compile graph."); sys.stdout.flush()
    if not vector_store: print("--- main.py (CONVERSATIONAL FINAL) ERROR: vector_store is None, cannot compile graph."); sys.stdout.flush()
    if not embeddings_model: print("--- main.py (CONVERSATIONAL FINAL) ERROR: embeddings_model is None, cannot compile graph."); sys.stdout.flush()
    graph = None
