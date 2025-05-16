# app.py (v2.1 - LangSmith Feedback Integration)
import sys
print("--- app.py (v2.1) IMPORTING ---"); sys.stdout.flush() # Make sure your running version reflects this
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client as LangSmithClient # Import LangSmith client
# from langchain_core.tracers.context import tracing_v2_enabled # May not be needed if run_id is in metadata

logger = logging.getLogger(__name__)

try:
    from main import graph as rag_graph
    if rag_graph is None:
         logger.error("--- app.py (v2.1) ERROR: RAG graph is None. Check main.py and set_env.py for errors.")
    logger.info("--- app.py (v2.1): Successfully imported RAG graph and related components.")
except ImportError as e:
    logger.error(f"--- app.py (v2.1) ERROR importing RAG graph or components: {e}", exc_info=True)
    rag_graph = None
except Exception as e:
    logger.error(f"--- app.py (v2.1) An unexpected error occurred during import: {e}", exc_info=True)
    rag_graph = None

app = Flask(__name__)
CORS(app)

# Initialize LangSmith client (it will use environment variables)
try:
    langsmith_client = LangSmithClient()
    logger.info("--- app.py (v2.1): LangSmith client initialized.")
except Exception as e:
    logger.error(f"--- app.py (v2.1): Failed to initialize LangSmith client: {e}. Feedback to LangSmith might fail.", exc_info=True)
    langsmith_client = None


@app.route('/api/ask', methods=['POST'])
def ask_rag_conversational():
    if rag_graph is None:
        logger.error("--- app.py (v2.1) /api/ask: RAG graph not initialized properly.")
        return jsonify({"error": "Backend RAG system not initialized. Check server logs."}), 500

    data = request.get_json()
    if not data:
        logger.warning("--- app.py (v2.1) /api/ask: Invalid request: No JSON body.")
        return jsonify({"error": "Invalid request. No JSON body provided."}), 400

    question = data.get('question')
    thread_id = data.get('thread_id') # Client should send this

    if not question:
        logger.warning("--- app.py (v2.1) /api/ask: Invalid request: 'question' not in JSON body.")
        return jsonify({"error": "Invalid request. Please provide a 'question'."}), 400
    
    if not thread_id:
        logger.warning("--- app.py (v2.1) /api/ask: 'thread_id' not provided by client.")
        return jsonify({"error": "Invalid request. Please provide a 'thread_id'."}), 400

    logger.info(f"--- app.py (v2.1) /api/ask: Received question for thread_id '{thread_id}': \"{question}\""); sys.stdout.flush()

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
                elif hasattr(last_message, 'id') and last_message.id: # This is the message ID, not the run ID.
                     # If we can't get a specific run_id for the AIMessage, we might have to rely on
                     # the LangSmith UI or more complex callback mechanisms to link feedback to the overarching graph run.
                     # For now, we'll pass what we can to the frontend.
                     logger.warning(f"--- app.py (v2.1) /api/ask: 'run_id' not in AIMessage.response_metadata. Using AIMessage.id ('{last_message.id}') as a potential reference if frontend needs it, but it's not the trace run_id for feedback.")
                     # To truly get the parent run_id, a callback handler added to the graph is more robust.
                     # For now, we'll make do. The feedback API in LangSmith is flexible.
                     pass # run_id_for_feedback remains None or its previous value

        logger.info(f"--- app.py (v2.1) /api/ask: Generated answer for thread_id '{thread_id}'. Run ID for feedback (if found): {run_id_for_feedback}"); sys.stdout.flush()
        return jsonify({
            "answer": final_ai_message_content,
            "question": question,
            "thread_id": thread_id,
            "run_id": run_id_for_feedback # Send this to the frontend
        })

    except Exception as e:
        logger.error(f"--- app.py (v2.1) /api/ask: Error during RAG invocation: {e}", exc_info=True); sys.stdout.flush()
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/api/log_feedback', methods=['POST'])
def log_feedback_to_langsmith():
    if not langsmith_client:
        logger.error("--- app.py (v2.1) /api/log_feedback: LangSmith client not available.")
        return jsonify({"error": "Feedback system (LangSmith client) not initialized."}), 500

    data = request.get_json()
    if not data:
        logger.warning("--- app.py (v2.1) /api/log_feedback: Invalid request: No data provided.")
        return jsonify({"error": "Invalid request. No data provided."}), 400

    question = data.get('question')
    answer = data.get('answer')
    feedback_status = data.get('feedback') # "correct" or "incorrect"
    run_id = data.get('run_id') # The run_id for the LLM call that produced the answer
    # thread_id = data.get('thread_id') # This is good for your own logging if needed

    if not all([question, answer, feedback_status]):
        logger.warning(f"--- app.py (v2.1) /api/log_feedback: Invalid feedback data (missing q, a, or f): {data}")
        return jsonify({"error": "Invalid feedback. Missing 'question', 'answer', or 'feedback'."}), 400

    logger.info(f"--- app.py (v2.1) FEEDBACK RECEIVED: Run ID: {run_id}, Status: {feedback_status.upper()} - Q: \"{question}\" - A: \"{answer}\""); sys.stdout.flush()

    try:
        # LangSmith feedback: score (1 for correct, 0 for incorrect) or a key
        # Using a clear key is often better.
        feedback_key = "user_rating" # A general key for user feedback
        # You can use score for a simple thumbs up/down, or value for more descriptive feedback
        score = 1.0 if feedback_status == "correct" else 0.0 # Binary score (float is also accepted)
        # Alternatively, for more nuance, or if you have more than two states:
        # value = feedback_status # e.g., "correct", "incorrect", "needs_improvement"

        if run_id: # IMPORTANT: Feedback is most useful when tied to a specific run
            langsmith_client.create_feedback(
                run_id=run_id,      # The ID of the trace/run (e.g., an LLM call or a chain run)
                key=feedback_key,   # A custom key for the feedback type
                score=score,        # Numerical score (e.g., 0 for incorrect, 1 for correct)
                # value=feedback_status, # Alternatively, use 'value' for non-numeric or more descriptive feedback
                comment=f"User marked as {feedback_status}. Q: {question}", # Optional: More context
                source="user_interaction" # Optional: Indicate where the feedback came from
            )
            logger.info(f"--- app.py (v2.1) /api/log_feedback: Feedback for run_id '{run_id}' (key: {feedback_key}, score: {score}) sent to LangSmith."); sys.stdout.flush()
            return jsonify({"message": "Feedback received and sent to LangSmith."}), 200
        else:
            # If no run_id, the feedback can't be directly linked to a specific trace via this API call.
            # You could create a new "orphan" run in LangSmith just for this feedback,
            # or log it to a different system, or just rely on your server logs.
            logger.warning(f"--- app.py (v2.1) /api/log_feedback: No run_id provided by frontend. Feedback Q: '{question}', A: '{answer}', Status: '{feedback_status}' logged locally but not directly to a specific LangSmith trace via API.");
            # Example of creating an "orphan" feedback event (less ideal but possible)
            # This creates a new run of type "feedback" in LangSmith
            # langsmith_client.create_run(
            #     name="Orphaned User Feedback",
            #     run_type="tool", # or "llm" or a custom type; "tool" might fit for a feedback event
            #     inputs={"question": question, "answer": answer, "user_feedback": feedback_status},
            #     outputs={"status": "logged_without_direct_trace_link"}
            # )
            return jsonify({"message": "Feedback received (no run_id from frontend, not directly linked to a LangSmith trace via API)."}), 202

    except Exception as e:
        logger.error(f"--- app.py (v2.1) /api/log_feedback: Error sending feedback to LangSmith: {e}", exc_info=True); sys.stdout.flush()
        return jsonify({"error": f"An error occurred while sending feedback to LangSmith: {e}"}), 500

@app.route('/')
def index():
    return render_template('index.html')