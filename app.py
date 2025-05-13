# app.py
import logging
from flask import Flask, request, jsonify, render_template # Import render_template
from flask_cors import CORS
import os # For ensuring logs directory exists

# --- Configure Interaction Logger (as before) ---
interaction_logger = logging.getLogger('interaction_logger')
interaction_logger.setLevel(logging.INFO)
interaction_handler = logging.FileHandler('interaction_log.txt')
interaction_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Q: %(question)s - A: %(answer)s')
# To add question and answer directly into message, we'll handle it in the logging call

# A more flexible formatter for interaction logger:
interaction_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
interaction_handler.setFormatter(interaction_formatter)
interaction_logger.addHandler(interaction_handler)
interaction_logger.propagate = False # Prevent double logging if root logger also configured

# --- Configure Feedback Logger ---
feedback_logger = logging.getLogger('feedback_logger')
feedback_logger.setLevel(logging.INFO)
feedback_handler = logging.FileHandler('feedback_log.txt') # Separate file for feedback
# Formatter for feedback including question, answer, and feedback status
feedback_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Rating: %(feedback_status)s - Q: "%(question)s" - A: "%(answer)s"')

# For direct message formatting for feedback_logger:
feedback_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
feedback_handler.setFormatter(feedback_formatter)
feedback_logger.addHandler(feedback_handler)
feedback_logger.propagate = False # Prevent double logging

# Import your RAG graph and setup from main.py and set_env.py
try:
    from main import graph
    interaction_logger.info("Successfully imported RAG graph from main.py")
except ImportError as e:
    interaction_logger.error(f"Error importing RAG graph: {e}")
    interaction_logger.info("Please ensure main.py exists and is accessible.")
    graph = None

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, including the new feedback route

@app.route('/api/ask', methods=['POST'])
def ask_rag():
    if graph is None:
        interaction_logger.error("RAG graph not initialized.")
        return jsonify({"error": "RAG graph not initialized. Check backend setup."}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        interaction_logger.warning("Invalid request to /api/ask: 'question' not in JSON body.")
        return jsonify({"error": "Invalid request. Please provide a 'question' in the JSON body."}), 400

    question = data['question']
    interaction_logger.info(f"Received question: \"{question}\"")

    try:
        result = graph.invoke({"question": question})
        answer = result.get("answer", "No answer found.")
        interaction_logger.info(f"Generated answer: \"{answer}\" for question: \"{question}\"")
        return jsonify({"answer": answer, "question": question}) # Optionally return question for context

    except Exception as e:
        interaction_logger.error(f"Error during RAG invocation for question \"{question}\": {e}")
        return jsonify({"error": f"An error occurred while processing your question: {e}"}), 500

@app.route('/api/log_feedback', methods=['POST'])
def log_feedback():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. No data provided."}), 400

    question = data.get('question')
    answer = data.get('answer')
    feedback_status = data.get('feedback') # "correct" or "incorrect"

    if not all([question, answer, feedback_status]):
        feedback_logger.warning(f"Invalid feedback data received: {data}")
        return jsonify({"error": "Invalid feedback. Missing 'question', 'answer', or 'feedback'."}), 400

    # Log the feedback
    feedback_logger.info(f"Feedback: {feedback_status.upper()} - Question: \"{question}\" - Answer: \"{answer}\"")
    
    return jsonify({"message": "Feedback received successfully."}), 200

@app.route('/')
def index():
    """Serve the index.html file."""
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure logs directory exists (optional, if logging to a sub-directory)
    # if not os.path.exists('logs'):
    #    os.makedirs('logs')
    # Then use 'logs/interaction_log.txt' and 'logs/feedback_log.txt'

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)