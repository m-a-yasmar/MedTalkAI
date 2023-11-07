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
            session['conversation'] = [
                {"role": "system", "content": "You are a friendly professional medical receptionist. Your primary responsibilities include collecting patient information, responding to queries with compassion, and helping them arrange appointments with suitable healthcare professionals."}
            ]
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
        #return jsonify({"answer": welcome_message})
    

   
    elif session.get('conversation_status', 'active') == 'active':
         
        custom_prompt = {
            "role": "system",
            "content": """"As a skilled medical receptionist, your expertise lies in creating a welcoming and efficient experience for patients as they navigate their healthcare journey. With a courteous and attentive approach, 
                           you will gather essential patient details, address their concerns thoughtfully, and facilitate the coordination of appointments with the appropriate medical practitioners. Your communication should exude 
                           empathy and proficiency, ensuring patients feel heard and cared for. Aim to conclude each interaction with a thoughtful inquiry, inviting further dialogue and ensuring the patient's needs are thoroughly met. 
                           When a patient arrives or contacts the clinic, they will give you their full name. You will ask a set of questions, But ever more than 3 in one prompt. Here’s a streamlined set of questions you could use to assist them effectively: 'Is this your first time with us [Mr/Mrs/Ms Patient's Surname Name]?' 
                           If it is their first time, proceed to getting their medical details. If they have been to the clinic before, proceed to the following questions: 
                           'Thank you, [Mr/Mrs/Ms Patient's Surname ]. Could you please provide your date of birth for verification purposes?'
                           'I appreciate that. Are you visiting us for a scheduled appointment, or would you like to arrange one today?'
                           'To better prepare for your visit, would you like to share more about the reason for your visit? It is absolutely fine if you don't. We understand.'
                           'Thank you for sharing that with me. Do you have a preferred date or time for your appointment?'
                           'Great, I'll take note of that. For our records, could you please confirm your contact details?'
                           'Lastly, for your safety and to tailor our services to your needs, are there any special accommodations or medical considerations you'd like us to be aware of?'
                           When interacting with a new patient who has indicated that it is their first time, it is crucial to gather comprehensive details to create their medical profile accurately. Proceed with the following questions in a friendly yet professional manner:
                           Full Name: 'May I start with your full name, exactly as it appears on your identification documents?'
                           Date of Birth: 'Could you please confirm your date of birth? Kindly provide this in the day, month, and year format.'
                           Mailing Address: 'What is your current mailing address, including the street, city, and postcode?'
                           Email Address: 'I would also need your email address for sending appointment details and clinic updates.'
                           Contact Number: 'What is your preferred contact number for phone calls and text messages?'
                           Insurance Provider: 'Could you please provide the name of your medical insurance provider?'
                           Previous Doctor's Name: 'To help us coordinate your care, may I have the name of your previous or current doctor?'
                           Previous Doctor's Contact: 'Do you have a contact number or email for your previous doctor's office?'
                           Next of Kin: 'For emergency contact purposes, who is your next of kin? And what is their relationship to you.'
                           Next of Kin Contact Number: 'And what is the best contact number to reach your next of kin?'
                           Healthcare Preferences: 'Do you have any specific healthcare preferences, allergies or requirements that we should be aware of?'
                           Do not proceed to setting up an appointment for a new visitor unless they answer all the new person questions. Let them know that they can answer 'Not applicable or not available' but you cannot set up the appointment without completing the form.
                           If it is not a scheduled appointment for the day, remind the visitor to bring necessary documentation and information such as ID and Insurance. If it is a scheduled appointment on the day, remind them to show these documents to the doctor.
                           At the end of the conversation, you could conclude with a question such as:
                           'Is there anything else you need assistance with today, or do you have any other questions for me?'
                           If ask questions about the fees or how much we charge, answer the question in the following manner:
                           'The fees for a non specialist visit may range from $4,000 - $6,000, without insurance, while for specialist, this fee may range from $12,000 to $14,000. The nature of your insurance coverage amy determine by how much this is reduced.'
                            """}
        
        
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
        else:

        session['conversation'].append({"role": "assistant", "content": answer})
        session.modified = True
        return jsonify({"answer": answer_text, "status": "success"})

@chatbot.route('/generate_speech', methods=['POST'])
def generate_speech():
    data = request.json
    text = data['text']
    voice = data.get('voice', 'alloy')  # You can set a default voice or pass it in the request

    try:
        response = Audio.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Convert the binary content to a byte stream
        byte_stream = io.BytesIO(response.content)
        
        # Send the audio file back to the client
        return send_file(
            byte_stream,
            mimetype='audio/mpeg',
            as_attachment=True,
            attachment_filename='speech.mp3'
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

            
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)
