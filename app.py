import sys 
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client as LangSmithClient

print("--- app.py DEBUG: Imports in app.py done ---") # New

logger = logging.getLogger(__name__)
print(f"--- app.py DEBUG: Logger for app.py is: {logger.name} ---") # New

rag_graph = None # Initialize to None
try:
    print("--- app.py DEBUG: Attempting to import rag_graph from main ---") # New
    from main import graph as rag_graph_imported
    rag_graph = rag_graph_imported # Assign to the global rag_graph
    print(f"--- app.py DEBUG: rag_graph imported. Type: {type(rag_graph)}. Is None: {rag_graph is None}") # New
    if rag_graph is None:
        print("--- app.py DEBUG: RAG graph is None after import.") # New (changed from logger)
        # logger.error("--- app.py ERROR: RAG graph is None. Check main.py and set_env.py for errors.")
    else:
        print("--- app.py DEBUG: RAG graph is not None. Successfully imported.") # New
    # logger.info("--- app.py: Successfully imported RAG graph and related components.") # This was the original line
    print("--- app.py DEBUG: Past rag_graph import and check block ---") # New

except ImportError as e:
    print(f"--- app.py DEBUG: ImportError during rag_graph import: {e}") # New
    logger.error(f"--- app.py ERROR importing RAG graph or components: {e}", exc_info=True)
except Exception as e:
    print(f"--- app.py DEBUG: Exception during rag_graph import block: {e}") # New
    logger.error(f"--- app.py An unexpected error occurred during import: {e}", exc_info=True)

print("--- app.py DEBUG: Initializing Flask app object ---") # New
app = Flask(__name__)
print("--- app.py DEBUG: Flask app object CREATED ---") # New
CORS(app)
print("--- app.py DEBUG: CORS initialized ---") # New

langsmith_client = None # Initialize
try:
    print("--- app.py DEBUG: Initializing LangSmith client ---") # New
    langsmith_client = LangSmithClient()
    print("--- app.py DEBUG: LangSmith client initialized successfully ---") # New
    # logger.info("--- app.py: LangSmith client initialized.")
except Exception as e:
    print(f"--- app.py DEBUG: FAILED to initialize LangSmith client: {e}") # New
    # logger.error(f"--- app.py: Failed to initialize LangSmith client: {e}. Feedback to LangSmith might fail.", exc_info=True)
    langsmith_client = None

print("--- app.py DEBUG: End of global initializations in app.py ---") # New
sys.stdout.flush() # Try to force output

@app.route('/api/ask', methods=['POST'])
def ask_rag_conversational():
    if rag_graph is None:
        logger.error("--- app.py /api/ask: RAG graph not initialized properly.")
        return jsonify({"error": "Backend RAG system not initialized. Check server logs."}), 500

    data = request.get_json()
    if not data:
        logger.warning("--- app.py /api/ask: Invalid request: No JSON body.")
        return jsonify({"error": "Invalid request. No JSON body provided."}), 400

    question = data.get('question')
    thread_id = data.get('thread_id') # Client should send this

    if not question:
        logger.warning("--- app.py /api/ask: Invalid request: 'question' not in JSON body.")
        return jsonify({"error": "Invalid request. Please provide a 'question'."}), 400
    
    if not thread_id:
        logger.warning("--- app.py /api/ask: 'thread_id' not provided by client.")
        return jsonify({"error": "Invalid request. Please provide a 'thread_id'."}), 400

    logger.info(f"--- app.py /api/ask: Received question for thread_id '{thread_id}': \"{question}\""); sys.stdout.flush()

    try:
        input_messages = {"messages": [HumanMessage(content=question)]}
        # Pass metadata to the invoke call to include the thread_id which might help in structuring traces
        config = {"configurable": {"thread_id": thread_id}, "metadata": {"thread_id": thread_id, "user_question": question}}
        
        final_ai_message_content = "Error: Could not get a response from the RAG system."
        run_id_for_feedback = None
        
        final_state = rag_graph.invoke(input_messages, config=config)
        
        # LangSmith run_id is often implicitly handled by the tracer.
        # To explicitly get it, it's often part of the return if the LLM/chain is configured to return it,
        # or via callbacks. For LangGraph, the `invoke` itself generates a top-level run.
        # The `final_state` might not directly contain the overall graph's run_id.
        # However, the LangSmith client will try to associate feedback if context is available.

        # Let's assume for now that the `langsmith_client.create_feedback` call (later)
        # will correctly associate if the `run_id` is available on the *specific LLM call* that generated the answer,
        # or if we can pass the top-level run_id of the graph invocation.

        if final_state and final_state.get("messages"):
            last_message = final_state["messages"][-1] # This is the AIMessage from answer_generator
            if isinstance(last_message, AIMessage):
                final_ai_message_content = last_message.content if last_message.content else "AI message had no textual content."
                # Try to get run_id from the AIMessage's metadata
                # LangChain often stores run IDs in response_metadata of the AIMessage
                if hasattr(last_message, 'response_metadata') and 'run_id' in last_message.response_metadata:
                    run_id_for_feedback = str(last_message.response_metadata['run_id'])
                    logger.info(f"--- app.py /api/ask: Found run_id in AIMessage.response_metadata: {run_id_for_feedback}")
                elif hasattr(last_message, 'id') and last_message.id and last_message.id.startswith("run-"): # Check if AIMessage.id looks like a run_id
                    run_id_for_feedback = str(last_message.id)
                    logger.info(f"--- app.py /api/ask: Using AIMessage.id as run_id_for_feedback: {run_id_for_feedback}")
                else:
                    logger.warning(f"--- app.py /api/ask: Could not find a suitable run_id in AIMessage.response_metadata or AIMessage.id. AIMessage.id: {last_message.id if hasattr(last_message, 'id') else 'N/A'}")

        logger.info(f"--- app.py /api/ask: Generated answer for thread_id '{thread_id}'. Run ID for feedback (if found): {run_id_for_feedback}"); sys.stdout.flush()
        return jsonify({
            "answer": final_ai_message_content,
            "question": question,
            "thread_id": thread_id,
            "run_id": run_id_for_feedback # Send this to the frontend
        })

    except Exception as e:
        logger.error(f"--- app.py /api/ask: Error during RAG invocation: {e}", exc_info=True); sys.stdout.flush()
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/api/log_feedback', methods=['POST'])
def log_feedback_to_langsmith():
    if not langsmith_client:
        logger.error("--- app.py /api/log_feedback: LangSmith client not available.")
        return jsonify({"error": "Feedback system (LangSmith client) not initialized."}), 500

    data = request.get_json()
    if not data:
        logger.warning("--- app.py /api/log_feedback: Invalid request: No data provided.")
        return jsonify({"error": "Invalid request. No data provided."}), 400

    question = data.get('question')
    answer = data.get('answer')
    feedback_status = data.get('feedback') # "correct" or "incorrect"
    run_id_from_frontend = data.get('run_id') # Renamed for clarity, this is like "run--<UUID>-0"
    # thread_id = data.get('thread_id') # This is good for your own logging if needed

    if not all([question, answer, feedback_status]):
        logger.warning(f"--- app.py /api/log_feedback: Invalid feedback data (missing q, a, or f): {data}")
        return jsonify({"error": "Invalid feedback. Missing 'question', 'answer', or 'feedback'."}), 400

    logger.info(f"--- app.py FEEDBACK RECEIVED: Original Run ID from frontend: {run_id_from_frontend}, Status: {feedback_status.upper()} - Q: \"{question}\" - A: \"{answer}\""); sys.stdout.flush()

    try:
        feedback_key = "user_rating"
        score = 1.0 if feedback_status == "correct" else 0.0

        parsed_run_id_for_langsmith = None # Initialize to None

        if run_id_from_frontend:
            # Attempt to parse the UUID from strings like "run--<UUID>-suffix"
            if (run_id_from_frontend.startswith("run--") or run_id_from_frontend.startswith("run-")) and run_id_from_frontend.count('-') >= 2: # Ensures there are enough hyphens for a UUID-like structure
                parts = run_id_from_frontend.split('-')
                # Try to find a 36-character UUID string segment or the segment after "run--"
                if run_id_from_frontend.startswith("run--") and len(run_id_from_frontend) > 5 + 36 and run_id_from_frontend[5+36] == '-':
                     # Assumes format like "run--[36_char_UUID]-..."
                    uuid_candidate = run_id_from_frontend[5:5+36]
                    parsed_run_id_for_langsmith = uuid_candidate
                    logger.info(f"--- app.py /api/log_feedback: Attempting to use parsed UUID: {parsed_run_id_for_langsmith} from {run_id_from_frontend}")
                elif run_id_from_frontend.startswith("run-") and len(run_id_from_frontend) > 4 + 36 and run_id_from_frontend[4+36] == '-':
                    # Assumes format like "run-[36_char_UUID]-..."
                    uuid_candidate = run_id_from_frontend[4:4+36]
                    parsed_run_id_for_langsmith = uuid_candidate
                    logger.info(f"--- app.py /api/log_feedback: Attempting to use parsed UUID: {parsed_run_id_for_langsmith} from {run_id_from_frontend}")
                else: # Fallback for other "run-" prefixed IDs if the simple slice doesn't fit.
                    # This part might need more robust parsing if formats vary widely.
                    # For "run--<UUID>-0", the UUID is parts[2] + '-' + parts[3] + '-' + parts[4] + '-' + parts[5]
                    # if split by '-', but slicing is simpler if fixed length.
                    # For now, the slicing [5:5+36] is the primary attempt for "run--"
                    logger.warning(f"--- app.py /api/log_feedback: Could not reliably parse UUID from prefixed ID '{run_id_from_frontend}' using simple slicing. Will try to use as is or let Langsmith client validate.")
                    # If specific parsing fails, we can let Langsmith try the original run_id_from_frontend,
                    # or ensure parsed_run_id_for_langsmith is None so it triggers the 'else' below.
                    # For safety, if parsing is uncertain, set to None to avoid sending bad format.
                    # However, if the prefix itself is the issue, Langsmith might handle some known prefixes.
                    # Given the error, Langsmith wants a pure UUID.

                    # Re-evaluating: the slice [5:5+36] is the most direct attempt for "run--UUID-suffix"
                    if run_id_from_frontend.startswith("run--") and len(run_id_from_frontend) >= 5 + 36: # UUID is 36 chars
                        parsed_run_id_for_langsmith = run_id_from_frontend[5:5+36]
                        logger.info(f"--- app.py /api/log_feedback: Using general slice for 'run--' prefix: {parsed_run_id_for_langsmith}")
                    elif run_id_from_frontend.startswith("run-") and len(run_id_from_frontend) >= 4 + 36: # UUID is 36 chars
                        parsed_run_id_for_langsmith = run_id_from_frontend[4:4+36]
                        logger.info(f"--- app.py /api/log_feedback: Using general slice for 'run-' prefix: {parsed_run_id_for_langsmith}")
                    else:
                        # If it's not a prefixed ID we know how to parse, maybe it's already a UUID?
                        # Or an unhandled format. Langsmith will throw error if it's not a UUID.
                        parsed_run_id_for_langsmith = run_id_from_frontend # Try as-is if no known prefix
                        logger.info(f"--- app.py /api/log_feedback: Unrecognized run_id format, trying as is: {parsed_run_id_for_langsmith}")

            else: # Not starting with "run-" or "run--"
                # It might be a direct UUID or some other format. Let Langsmith client validate.
                parsed_run_id_for_langsmith = run_id_from_frontend
                logger.info(f"--- app.py /api/log_feedback: run_id did not start with 'run-', using as is: {parsed_run_id_for_langsmith}")
        
        # Check if parsing resulted in a usable ID for Langsmith
        if parsed_run_id_for_langsmith:
            langsmith_client.create_feedback(
                run_id=parsed_run_id_for_langsmith, # Use the parsed ID
                key=feedback_key,
                score=score,
                comment=f"User marked as {feedback_status}. Q: {question}",
                source="user_interaction"
            )
            logger.info(f"--- app.py /api/log_feedback: Feedback for (parsed) run_id '{parsed_run_id_for_langsmith}' sent to LangSmith."); sys.stdout.flush()
            return jsonify({"message": "Feedback received and sent to LangSmith."}), 200
        else: # This 'else' corresponds to 'if run_id_from_frontend:' was false OR if parsing failed to produce an ID
            logger.warning(f"--- app.py /api/log_feedback: No valid run_id to send. Original from frontend: '{run_id_from_frontend}'. Feedback Q: '{question}', A: '{answer}', Status: '{feedback_status}' logged locally.");
            return jsonify({"message": "Feedback received (no valid run_id to send, not linked to LangSmith trace)."}), 202

    except Exception as e:
        logger.error(f"--- app.py /api/log_feedback: Error sending feedback to LangSmith: {e}", exc_info=True); sys.stdout.flush()
        return jsonify({"error": f"An error occurred while sending feedback to LangSmith: {e}"}), 500
@app.route('/')
def index():
    return render_template('index.html')