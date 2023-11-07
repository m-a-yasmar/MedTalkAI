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
import io
from flask import send_file
from pydub import AudioSegment 


logging.basicConfig(level=logging.DEBUG)

chatbot = Flask(__name__)

from flask_cors import CORS # for CORS

CORS(chatbot)

chatbot.secret_key = 'actual_voice_secret_medical_app1'  # Replace with your secret key
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name
    try:
        with open(temp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcribed_text = transcript['text']
            session['transcribed_text'] = transcribed_text
            return ask_endpoint(transcribed_text)
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
@limiter.limit("6 per minute; 20 per 10 minutes; 30 per hour")
def custom_limit_request_error():
    return jsonify({"message": "Too many requests, please try again later"}), 429

@chatbot.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    return ask_endpoint(query)

def ask_endpoint(query):
    threshold = 0.9
    max_tokens = 50
    tokens = query.split()
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
        session.clear()
        session['cleared'] = True
        session['returning_user'] = False
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
            return_message = "Hello and a warm welcome! I'm Sam, your AI medical receptionist here to assist you. Before we proceed may I have your full name please?"
            session['awaiting_decision'] = False
            session['conversation_status'] = 'active'
        session['conversation'].append({"role": "assistant", "content": return_message})
        return jsonify({"answer": return_message})
    elif session.get('conversation_status', 'new') == 'new':
        welcome_message = "Hello and a warm welcome! I'm Sam, your medical receptionist here to assist you."
        session['conversation'].append({"role": "assistant", "content": welcome_message})
        session['conversation_status'] = 'active'
        return jsonify({"answer": welcome_message})
    elif session.get('conversation_status', 'active') == 'active':
        custom_prompt = {
            "role": "system",
            "content": "custom_prompt_content"
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
        answer = "I'm sorry, I couldn't understand the question."
        if response.status_code == 200:
            answer_text = response.json()['choices'][0]['message']['content']
            forbidden_phrases = ["I am a model trained", "As an AI model", "My training data includes", "ChatGPT","OpenAI"]
            for phrase in forbidden_phrases:
                answer = answer.replace(phrase, "")
            session['conversation'].append({"role": "assistant", "content": answer_text})
            session.modified = True
            return jsonify({"answer": answer_text, "status": "success"})
        else:
            error_message = "An error occurred while processing your request."
            session['conversation'].append({"role": "assistant", "content": error_message})
            return jsonify({"answer": error_message, "status": "error"})
        



@chatbot.route('/generate_speech', methods=['POST'])
def generate_speech():
    data = request.json
    text = data['text']
    voice = data.get('voice', 'alloy')  # You can set a default voice or pass it in the request

    try:
        response = requests.post(
            "https://api.openai.com/v1/engines/davinci/tts-1",
            headers={
                "Authorization": f"Bearer {os.environ.get('MEDTALK_API_KEY')}",
               
            },
            json={
                "text": text,
                "voice": voice
            }
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            # Convert the MP3 content to WAV
            mp3_audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
            wav_io = io.BytesIO()
            mp3_audio.export(wav_io, format="wav")
            wav_io.seek(0)  # Go to the beginning of the stream

            # Send the WAV audio file back to the client
            return send_file(
                wav_io,
                mimetype='audio/wav',
                as_attachment=True,
                attachment_filename='speech.wav'
            )
        else:
            # Handle the error if the API call failed
            return jsonify({"status": "error", "message": "Failed to generate speech"}), response.status_code

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
            
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)
