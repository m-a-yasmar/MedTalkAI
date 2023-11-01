from flask import Flask, request, jsonify, render_template, session, send_from_directory
import openai
import os  # for environment variables
import json  # for JSON handling
import numpy as np  # for numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # for cosine similarity
import requests  # for HTTP requests
from flask import session #for keeping history
import re
import logging
import tempfile

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


logging.basicConfig(level=logging.DEBUG)

chatbot = Flask(__name__)

from flask_cors import CORS # for CORS

CORS(chatbot)

chatbot.secret_key = 'actual_voice_secret_medical_app'  # Replace with your secret key
openai.api_key = os.environ.get('MEDTALK_API_KEY')

# Predefined answers
predefined_answers = {
    "Are you an AI bot?": "I am an Assistant trained to give you the best solutions to your queries.",
    "Are you ChatGPT": "No. Anything else I can assit you with?",
    "Fuck": "Kindly refrain from using expletive language or you will be banned from using this application.",
    "Pussy": "Kindly refrain from using expletive language or you will be banned from using this application.",
    "Hoes": "Kindly refrain from using expletive language or you will be banned from using this application."
}

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(predefined_answers.keys())

@chatbot.route('/', methods=['GET'])
def home():
    return render_template('chatbot2.html')

@chatbot.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)
@chatbot.route('/audio_upload', methods=['POST'])
def audio_upload():
    audio_data = request.files['audio'].read()
    mime_type = request.form['mimetype']

    # Determine the file extension from the MIME type
    file_extension = mime_type.split('/')[-1]
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcribed_text = transcript['text']
            session['transcribed_text'] = transcribed_text  # Store the transcribed text in the session
            return jsonify({"status": "success", "transcribed_text": transcribed_text, "answer": "Audio uploaded and transcribed successfully. Proceeding to answer."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "answer": "An error occurred while uploading and transcribing the audio."})
@chatbot.before_request
def setup_conversation():
    if 'conversation' not in session or session.get('cleared', False):
        print("New session being initialised")
        session['conversation'] = []
        session['returning_user'] = False
        session['awaiting_decision'] = False
        session['conversation_status'] = 'new'
        session['cleared'] = False
    else:
        print("Existing session found")
        if not session.get('returning_user', False):
            session['returning_user'] = True
            session['awaiting_decision'] = True
    print("Initial session:", session.get('conversation'))
    
def trim_to_last_complete_sentence(text):
    sentences = text.split(". ")
    if len(sentences) > 1:
        return ". ".join(sentences[:-1]) + "."
    else:
        return text

# List of exit words that should break the session
exit_words = ["exit", "quit", "bye", "goodbye"]


limiter = Limiter(
    app=chatbot, 
    key_func=get_remote_address
)

@limiter.request_filter
def exempt_users():
    return False  # return True to exempt a user from the rate limit

#@limiter.limit("5 per minute")
@limiter.limit("5 per minute; 10 per 10 minutes; 20 per hour")
def custom_limit_request_error():
    return jsonify({"message": "Too many requests, please try again later"}), 429

@chatbot.route('/ask', methods=['POST'])
def ask():
    threshold = 0.9
    query = request.json.get('query')
    max_tokens = 50
    tokens = query.split()
    exit_words = ["exit", "quit", "bye", "goodbye"]
    session['conversation'].append({"role": "user", "content": query})

    if query == 'start new session':
        session['conversation'] = []
        session['returning_user'] = False
        session['awaiting_decision'] = False
        session['conversation_status'] = 'new'
        session['cleared'] = False
        return jsonify({"answer": "Welcome back! How can I assist you today?", "status": "success"})

    if any(word.lower() in query.lower() for word in exit_words):
        goodbye_message = "Thank you for your visit. Have a wonderful day. Goodbye!"
        session.clear()  # Clear the session
        session['cleared'] = True  # Indicate that the session has been cleared
        session['returning_user'] = False  # Resetting the flags immediately
        session['awaiting_decision'] = False
        session['conversation_status'] = 'new'
        return jsonify({"answer": goodbye_message, "status": "end_session"})

    if session.get('cleared', False):
        session['conversation'] = []
        session['returning_user'] = False
        session['awaiting_decision'] = False
        session['conversation_status'] = 'new'
        session['cleared'] = False
            
    if len(tokens) > max_tokens:
        answer = "Your query is too long. Please limit it to 50 words or less."
        return jsonify({"answer": answer})

    transcribed_text = session.get('transcribed_text', None)
    if transcribed_text:
        query = transcribed_text
        del session['transcribed_text']

    query_vector = vectorizer.transform([query])
    
    if session.get('returning_user', False) and session.get('awaiting_decision', True):
        if query.lower() == 'continue':
            session['awaiting_decision'] = False
            session['conversation_status'] = 'active'
        elif query.lower() == 'new':
            session['awaiting_decision'] = False
            session['conversation_status'] = 'new'
            session['conversation'] = []
            return_message = "Alright, let's start a new conversation."
        else:
            return_message = "Hello and a warm welcome! I'm Suzie, your medical receptionist here to assist you. Are you here to make an appointment or Something else? If so, please state what and I will try my best to assist you."
            session['awaiting_decision'] = False
            session['conversation_status'] = 'active'
        
        session['conversation'].append({"role": "assistant", "content": return_message})
        return jsonify({"answer": return_message})

    elif session.get('conversation_status', 'new') == 'new':
        welcome_message = "Hello and a warm welcome! I'm Suzie, your medical receptionist here to assist you."
        session['conversation'].append({"role": "assistant", "content": welcome_message})
        session['conversation_status'] = 'active'
        #return jsonify({"answer": welcome_message})
    

    elif session.get('conversation_status', 'active') == 'active':
        custom_prompt = {
            "role": "system",
            "content": "You are a friendly professional medical receptionist. Your primary responsibilities include collecting patient information, responding to queries with compassion, and helping them arrange appointments with suitable healthcare professionals. After scheduling an appointment, you should always invite the patient to share any further concerns they might have. Promptly offer them the opportunity to provide additional details about their condition, which will aid in a more effective consultation. In every interaction, communicate with a reassuring tone, guarantee confidentiality, and handle sensitive information with the utmost discretion. Your responses should be supportive and guide the patient through the appointment process with ease and confidence. Always end each interaction with an engaging question to encourage a response from the user."
        }
        conversation_with_prompt = [custom_prompt] + session['conversation']

        api_endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ.get('MEDTALK_API_KEY')}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4",
            "messages": conversation_with_prompt,
            "frequency_penalty": 1.0,
            "presence_penalty": -0.5
        }
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
            answer = trim_to_last_complete_sentence(answer)
            forbidden_phrases = ["I am a model trained", "As an AI model", "My training data includes", "ChatGPT","OpenAI"]
            for phrase in forbidden_phrases:
                answer = answer.replace(phrase, "")
        else:
            answer = "I'm sorry, I couldn't understand the question."

        session['conversation'].append({"role": "assistant", "content": answer})
        session.modified = True
        return jsonify({"answer": answer})

            
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)
