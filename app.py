# app.py (v2 - Conversational RAG)
import sys
print("--- app.py (v2) IMPORTING ---"); sys.stdout.flush()
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import datetime # For feedback timestamp
from uuid import uuid4 # For generating thread_ids if needed

# Langchain message types
from langchain_core.messages import HumanMessage, AIMessage 

# --- Configure Unified Console Logger ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your RAG graph
try:
    from main import graph as rag_graph # Renamed to avoid conflict
    if rag_graph is None:
         logger.error("--- app.py (v2) ERROR: RAG graph is None. Check main.py and set_env.py for errors.")
    logger.info("--- app.py (v2): Successfully imported RAG graph and related components.")
except ImportError as e:
    logger.error(f"--- app.py (v2) ERROR importing RAG graph or components: {e}")
    rag_graph = None
except Exception as e:
    logger.error(f"--- app.py (v2) An unexpected error occurred during import: {e}")
    rag_graph = None

app = Flask(__name__)
CORS(app)

# In-memory store for chat histories (for simplicity in this example, ideally use a DB for persistence if not relying solely on LangGraph checkpointer for all history needs)
# With LangGraph's MemorySaver, the history is managed by the checkpointer per thread_id.
# The client will typically display its own version of history.

@app.route('/api/ask', methods=['POST'])
def ask_rag_conversational():
    if rag_graph is None:
        logger.error("--- app.py (v2) /api/ask: RAG graph not initialized properly.")
        return jsonify({"error": "Backend RAG system not initialized. Check server logs."}), 500

    data = request.get_json()
    if not data:
        logger.warning("--- app.py (v2) /api/ask: Invalid request: No JSON body.")
        return jsonify({"error": "Invalid request. No JSON body provided."}), 400

    question = data.get('question')
    thread_id = data.get('thread_id') # Client should send this

    if not question:
        logger.warning("--- app.py (v2) /api/ask: Invalid request: 'question' not in JSON body.")
        return jsonify({"error": "Invalid request. Please provide a 'question'."}), 400
    
    if not thread_id:
        # If client doesn't send a thread_id, we can generate one for this session.
        # However, for true conversational continuity, client should generate on "New Chat" and persist it.
        # For now, let's require it or make a new one each time if not provided (less ideal for history)
        logger.warning("--- app.py (v2) /api/ask: 'thread_id' not provided by client. A new one might be created by LangGraph if not managed.")
        # If you want to enforce client sending thread_id:
        # return jsonify({"error": "Invalid request. Please provide a 'thread_id'."}), 400
        # For robust behavior, ensure client always sends a thread_id.
        # If a new thread_id is needed (e.g. client starts a 'new chat'), client generates it.


    logger.info(f"--- app.py (v2) /api/ask: Received question for thread_id '{thread_id}': \"{question}\""); sys.stdout.flush()

    try:
        # The input to the graph is a list of messages. For a new turn, it's just the new HumanMessage.
        # The checkpointer associated with the graph will load the history for the given thread_id.
        input_messages = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": thread_id}}
        
        final_ai_message_content = "Error: Could not get a response from the RAG system."
        
        # Using .invoke() gets the final state. We need the last AI message.
        # The tutorial uses .stream() to show intermediate steps.
        # For a simple request-response, .invoke() is fine, then we extract the last message.
        final_state = rag_graph.invoke(input_messages, config=config)
        
        # The final state 'messages' list contains all messages in the thread after this turn.
        # The last one should be the AI's response.
        if final_state and final_state.get("messages"):
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                final_ai_message_content = last_message.content
            elif isinstance(last_message, AIMessage) and not last_message.content and last_message.tool_calls:
                final_ai_message_content = "The AI decided to use a tool but didn't produce a final text response in this step."
            elif isinstance(last_message, ToolMessage): # Should not be the final message if graph is correct
                 final_ai_message_content = "A tool was called, but the final generation step might be missing or failed."
            else: # Other message type or unexpected state
                final_ai_message_content = f"Unexpected final message type: {type(last_message)}. Please check backend logic."

        else: # Should not happen if graph runs to END
            final_ai_message_content = "RAG system did not return a final message state."


        logger.info(f"--- app.py (v2) /api/ask: Generated answer for thread_id '{thread_id}': \"{final_ai_message_content}\""); sys.stdout.flush()
        return jsonify({"answer": final_ai_message_content, "question": question, "thread_id": thread_id})

    except Exception as e:
        logger.error(f"--- app.py (v2) /api/ask: Error during RAG invocation for question \"{question}\", thread_id '{thread_id}': {e}", exc_info=True); sys.stdout.flush()
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/api/log_feedback', methods=['POST'])
def log_feedback():
    data = request.get_json()
    if not data:
        logger.warning("--- app.py (v2) /api/log_feedback: Invalid request: No data provided.")
        return jsonify({"error": "Invalid request. No data provided."}), 400

    question = data.get('question')
    answer = data.get('answer')
    feedback_status = data.get('feedback')
    # thread_id = data.get('thread_id') # Optionally log thread_id with feedback

    if not all([question, answer, feedback_status]):
        logger.warning(f"--- app.py (v2) /api/log_feedback: Invalid feedback data: {data}")
        return jsonify({"error": "Invalid feedback. Missing 'question', 'answer', or 'feedback'."}), 400

    logger.info(f"--- app.py (v2) FEEDBACK: Status: {feedback_status.upper()} - Question: \"{question}\" - Answer: \"{answer}\""); sys.stdout.flush()
    # Here you would typically store this feedback in a database (e.g., MongoDB as discussed)
    # For example:
    # feedback_entry = {"question": question, "answer": answer, "feedback": feedback_status, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
    # if feedback_collection: feedback_collection.insert_one(feedback_entry)
    
    return jsonify({"message": "Feedback received successfully."}), 200

@app.route('/')
def index():
    """Serve the index.html file."""
    return render_template('index.html')

# Gunicorn runs the app, so no app.run() here for production
