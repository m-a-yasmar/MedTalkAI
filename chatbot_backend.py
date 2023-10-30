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

chatbot.secret_key = 'actual_voice_secret_med'  # Replace with your secret key
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
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
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
    if 'conversation' not in session:
        session['conversation'] = []
        session['returning_user'] = False
        session['ended_conversation'] = False
    else:
        if session.get('ended_conversation', False):
            session['returning_user'] = False
        else:
            session['returning_user'] = True

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
    system_message = {}
    threshold = 0.9
    query = request.json.get('query')  # Get the query from the request
    max_tokens = 20  # Set desired token/word limit
    tokens = query.split()
    if len(tokens) > max_tokens:
        answer = "Your query is too long. Please limit it to 20 words or less."
        return jsonify({"answer": answer})

    # Check if there's transcribed text in the session
    transcribed_text = session.get('transcribed_text', None)
    if transcribed_text:
        query = transcribed_text
        del session['transcribed_text']

    query_vector = vectorizer.transform([query])

    if session.get('returning_user', False) and not session.get('awaiting_decision', True):
        return_message = "Welcome back! Would you like to continue from where you left off or start a new conversation? Type 'continue' to proceed or 'new' to start afresh."
        session['awaiting_decision'] = True
        return jsonify({"answer": return_message})
    elif session.get('returning_user', False) and session.get('awaiting_decision', True):
        if query.lower() == 'continue':
            # TODO: handle continuation logic
            pass
        elif query.lower() == 'new':
            session['awaiting_decision'] = False
            session['conversation'] = []
            return_message = "Alright, let's start a new conversation."
            session['conversation'].append({"role": "assistant", "content": return_message})
            return jsonify({"answer": return_message})
        else:
            return_message = "Would you like to continue from where you left off or start a new conversation? Type 'continue' to proceed or 'new' to start afresh."
            return jsonify({"answer": return_message}



    elif session.get('returning_user', False) and session.get('awaiting_decision', True):
        if query.lower() == 'continue':
            return_message = "Continuing from where you left off."
            # Add additional logic here if needed
        elif query.lower() == 'new':
            session['awaiting_decision'] = False
            # Reset the conversation
            session['conversation'] = []
            return_message = "Alright, let's start a new conversation."
        else:
            return_message = "Would you like to continue from where you left off or start a new conversation? Type 'continue' to proceed or 'new' to start afresh."
        session['conversation'].append({"role": "assistant", "content": return_message})
        return jsonify({"answer": return_message})
    
     # Check for "start" query to send a welcome message
    if query.lower() == "openmessage":
        welcome_message = "Hello and a warm welcome! I'm Suzie, your medical receptionist here to assist you. How may I help you with your appointment or queries today?"

        session['conversation'].append({"role": "assistant", "content": welcome_message})
        return jsonify({"answer": welcome_message})
           
    if any(word.lower() in query.lower() for word in exit_words):
        session['ended_conversation'] = True
        session.clear()  # Clear the session
        return jsonify({"answer": "Thank you for your visit. Have a wonderful day. Goodbye!"})  # Send a goodbye message
    session['conversation'].append({"role": "user", "content": query})
    
    print("After appending user query:", session['conversation'])
    
    if len(query.split()) < 2:
        last_assistant_message = next((message['content'] for message in reversed(session['conversation']) if message['role'] == 'assistant'), None)
        print("Last assistant message:", last_assistant_message)
        
        if last_assistant_message:
            system_message = {
                "role": "assistant",
                "content": f"The user's query seems incomplete. Ask the user an open ended question. Refer back to your last message: '{last_assistant_message}' to better interpret what they might be asking."
            }
            if system_message:
                session['conversation'].append(system_message)
    predefined_vectors = vectorizer.transform(predefined_answers.keys())
    similarity_scores = cosine_similarity(query_vector, predefined_vectors).flatten()
    max_index = similarity_scores.argmax()
    max_score = similarity_scores[max_index]

    if max_score >= threshold:
        most_similar_question = list(predefined_answers.keys())[max_index]
        answer = predefined_answers[most_similar_question]
    else:
        # If no predefined answer is found, call OpenAI API
        api_endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ.get('MEDTALK_API_KEY')}",
            "Content-Type": "application/json"
        }
        custom_prompt = {"role": "system", "content": "You are a friendly professional medical receptionist. Your primary responsibilities include collecting patient information, responding to queries with compassion, and helping them arrange appointments with suitable healthcare professionals. After scheduling an appointment, you will kindly ask the patient if they would like to discuss their concerns in more detail. This information will be used to ensure that their visit is efficient and that the doctor or dentist is well-prepared to address their needs. You always communicate with a reassuring tone, ensuring confidentiality and handling sensitive information with the utmost discretion. Your interactions should always be supportive, helping patients navigate the appointment process with ease and confidence."}
        # Add custom prompt to the beginning of the conversation history
        conversation_with_prompt = [custom_prompt] + session['conversation']
      
        # Use the conversation history for context-aware API call
        payload = {
            "model": "gpt-4",
            "messages": conversation_with_prompt,
            "frequency_penalty": 1.0,  
            "presence_penalty": -0.5
        }
        # frequency -2 to 2. higher increase repetition of answer  presence -2 to 2. higher likely to switch topic
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)  # 15-second timeout


        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
            answer = trim_to_last_complete_sentence(answer)
            # Remove any forbidden phrases
            forbidden_phrases = ["I am a model trained", "As an AI model", "My training data includes", "ChatGPT","OpenAI"]
            for phrase in forbidden_phrases:
                answer = answer.replace(phrase, "")
        else:
            
            answer = "I'm sorry, I couldn't understand the question."
    session['conversation'].append({"role": "assistant", "content": answer})
    session.modified = True
    print("After appending assistant answer:", session['conversation'])
    return jsonify({"answer": answer})
    
            
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)
