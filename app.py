import os
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from dotenv import load_dotenv
from datetime import datetime
import uuid
import time

# Import the LegalChatbot class from your existing code
from chatbot import LegalChatbot

# Load environment variables
load_dotenv(".env")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

# Configure server-side session
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_FILE_DIR"] = "flask_session"
app.config["SESSION_USE_SIGNER"] = True
Session(app)

# Create a dictionary to store chatbot instances for each session
chatbot_instances = {}

# Function to get or create chatbot instance for a session
def get_chatbot_instance(session_id):
    if session_id not in chatbot_instances:
        # Initialize a new chatbot instance
        chatbot_instances[session_id] = LegalChatbot(
            document_path=os.getenv("DOCUMENT_PATH", "litigation.txt"),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "litigation_faiss_index")
        )
    return chatbot_instances[session_id]

@app.route('/')
def home():
    # Ensure session has an ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        # Get data from request
        data = request.json
        query = data.get('query', '')
        
        if not query.strip():
            return jsonify({"error": "Empty query"}), 400
        
        # Get chatbot instance for this session
        chatbot = get_chatbot_instance(session_id)
        
        # Send query to chatbot
        start_time = time.time()
        response = chatbot.chat(query)
        end_time = time.time()
        
        # Get the conversation history
        history = chatbot.get_conversation_history()
        
        return jsonify({
            "response": response,
            "conversation_history": history,
            "processing_time": round(end_time - start_time, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        # Get session ID
        session_id = session.get('session_id')
        
        if not session_id or session_id not in chatbot_instances:
            return jsonify({"history": []})
        
        # Get chatbot instance
        chatbot = chatbot_instances[session_id]
        
        # Get the conversation history
        history = chatbot.get_conversation_history()
        
        return jsonify({"history": history})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    try:
        # Get session ID
        session_id = session.get('session_id')
        
        if session_id and session_id in chatbot_instances:
            # Clear the conversation history
            chatbot = chatbot_instances[session_id]
            chatbot.clear_conversation()
        
        return jsonify({"status": "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a function to clean up old sessions and their chatbot instances
@app.before_request
def cleanup_sessions():
    now = time.time()
    # Clean up sessions older than 2 hours
    sessions_to_remove = []
    for session_id in chatbot_instances:
        if 'last_access' not in session or now - session.get('last_access', 0) > 7200:  # 2 hours
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        if session_id in chatbot_instances:
            del chatbot_instances[session_id]
    
    # Update last access time
    session['last_access'] = now

if __name__ == '__main__':
    # Create the sessions directory if it doesn't exist
    os.makedirs("flask_session", exist_ok=True)
    
    # Run the Flask app
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")