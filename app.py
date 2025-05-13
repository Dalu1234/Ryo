# app.py
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# --- Configure Unified Console Logger ---
# All logs will go to stdout/stderr, which Render will capture.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a common logger

# Import your RAG graph and setup
try:
    from main import graph # graph should be initialized in main.py after set_env
    from set_env import vector_store # To check if it's available
    if vector_store is None:
        logger.error("Vector store not available. Check Qdrant connection in set_env.py.")
    if graph is None: # main.py might not explicitly set graph to None on error
         logger.error("RAG graph is None. Check main.py and set_env.py for errors.")
    logger.info("Successfully imported RAG graph and related components.")
except ImportError as e:
    logger.error(f"Error importing RAG graph or components: {e}")
    logger.info("Please ensure main.py and set_env.py exist and are accessible, and Qdrant is configured.")
    graph = None
    vector_store = None
except Exception as e:
    logger.error(f"An unexpected error occurred during import: {e}")
    graph = None
    vector_store = None


app = Flask(__name__)
CORS(app)

@app.route('/api/ask', methods=['POST'])
def ask_rag():
    if graph is None or vector_store is None:
        logger.error("RAG graph or vector_store not initialized properly.")
        return jsonify({"error": "Backend RAG system not initialized. Check server logs."}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        logger.warning("Invalid request to /api/ask: 'question' not in JSON body.")
        return jsonify({"error": "Invalid request. Please provide a 'question' in the JSON body."}), 400

    question = data['question']
    logger.info(f"Received question for /api/ask: \"{question}\"")

    try:
        # Ensure the graph uses the potentially updated vector_store from set_env
        # If 'graph' is a compiled LangGraph object, it should use the 'vector_store'
        # that was in scope when its components (like 'retrieve') were defined.
        # Re-compiling or ensuring components have access to the latest vector_store might be needed
        # if vector_store could change after initial graph compilation.
        # For this setup, main.py should define 'graph' after 'vector_store' is set in set_env.py.

        result = graph.invoke({"question": question})
        answer = result.get("answer", "No answer found.")
        logger.info(f"Generated answer: \"{answer}\" for question: \"{question}\"")
        return jsonify({"answer": answer, "question": question})

    except Exception as e:
        logger.error(f"Error during RAG invocation for question \"{question}\": {e}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing your question: {e}"}), 500

@app.route('/api/log_feedback', methods=['POST'])
def log_feedback():
    data = request.get_json()
    if not data:
        logger.warning("Invalid request to /api/log_feedback: No data provided.")
        return jsonify({"error": "Invalid request. No data provided."}), 400

    question = data.get('question')
    answer = data.get('answer')
    feedback_status = data.get('feedback')

    if not all([question, answer, feedback_status]):
        logger.warning(f"Invalid feedback data received: {data}")
        return jsonify({"error": "Invalid feedback. Missing 'question', 'answer', or 'feedback'."}), 400

    # Log the feedback to console (Render will pick this up)
    # For persistent feedback logging, you'd need a database or external logging service.
    logger.info(f"FEEDBACK: Status: {feedback_status.upper()} - Question: \"{question}\" - Answer: \"{answer}\"")

    return jsonify({"message": "Feedback received successfully."}), 200

@app.route('/')
def index():
    """Serve the index.html file."""
    return render_template('index.html')

# Remove the if __name__ == '__main__': block for Gunicorn
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False) # This line is for local dev server