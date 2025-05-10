# app.py
from flask import Flask, request, jsonify, render_template # Import render_template
from flask_cors import CORS

# Import your RAG graph and setup from main.py and set_env.py
try:
    from main import graph
    print("Successfully imported RAG graph from main.py")
except ImportError as e:
    print(f"Error importing RAG graph: {e}")
    print("Please ensure main.py exists and is accessible.")
    graph = None # Set graph to None if import fails

app = Flask(__name__)
CORS(app)

@app.route('/api/ask', methods=['POST'])
def ask_rag():
    """
    API endpoint to receive a question and return an answer from the RAG system.
    Expects a JSON payload with a 'question' key.
    """
    if graph is None:
        return jsonify({"error": "RAG graph not initialized. Check backend setup."}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid request. Please provide a 'question' in the JSON body."}), 400

    question = data['question']
    print(f"Received question: {question}")

    try:
        result = graph.invoke({"question": question})
        answer = result.get("answer", "No answer found.")
        print(f"Generated answer: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        return jsonify({"error": f"An error occurred while processing your question: {e}"}), 500

@app.route('/')
def index():
    """Serve the index.html file."""
    return render_template('index.html') # Use render_template to serve index.html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)