from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from flask_cors import CORS
import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from deepface import DeepFace
import json

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# Enable CORS for React frontend on port 3000
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})

# Configure session management
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Replace with a secure key, e.g., os.urandom(16).hex()
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "image_chat_collection"

try:
    collection = chroma_client.get_collection(name=collection_name)
except:
    collection = chroma_client.create_collection(name=collection_name)

def populate_chroma_db():
    knowledge_base = {
        "saliya": {
            "info": "I’m Saliya Withana, the CEO of Enfection. We create powerful and impactful campaigns and provide creative performance marketing solutions to our belief network, which drives our growth. It is their faith in us that has allowed us to think big and propel us forward in our growth trajectory. Our work has enabled brands to be relevant in ever-evolving markets across the globe while being engaged in the daily lives of the consumer.",
            "role": "Chief Executive Officer",
            "company": "Enfection",
            "welcome": "We are thrilled to have you join our team and embark on this exciting journey together. At Enfection, we believe that our people are at the heart of our success, and your unique talents and contributions play a vital role in our shared growth.",
            "who_we_are": "We started our journey in 2017 with the aim of providing comprehensive marketing solutions to our customers. We partner with brands to achieve meaningful progress, bringing value to our belief network through innovative, forward-thinking 360 solutions.",
            "values": "Integrity, Collaboration, Innovation, Transparency, Continuous Learning, Wellbeing, Client-Centric Approach, Adaptability, Team Collaboration, Work-Life Balance",
            "leadership": "Saliya Withange (CEO), Lahiru Halkewela (COO), Shezri Junaid (CSO), Rasanga Abhishek (CGO), Dhana Perera (Head of People and Culture), and others.",
            "policies": "Confidentiality: Do not disclose company info. Working Hours: 9 AM - 6 PM with flexibility. Leave: 7 days casual, 14 days annual (2nd year), 7 days medical (2nd year).",
            "fun_activities": "Houses: Sterling, Don, Peggy, Pete. Events: New Year Celebrations, Annual Trip, Sports Day, Secret Santa."
        },
        "dhana": {
            "info": "I’m Dhana Perera, the Head of People and Culture at Enfection. We create powerful and impactful campaigns and provide creative performance marketing solutions to our belief network, which drives our growth. It is their faith in us that has allowed us to think big and propel us forward in our growth trajectory. Our work has enabled brands to be relevant in ever-evolving markets across the globe while being engaged in the daily lives of the consumer.",
            "role": "Head of People and Culture",
            "company": "Enfection",
            "email": "dhana@enfection.com",
            "welcome": "We are thrilled to have you join our team and embark on this exciting journey together. At Enfection, we believe that our people are at the heart of our success, and your unique talents and contributions play a vital role in our shared growth.",
            "who_we_are": "We started our journey in 2017 with the aim of providing comprehensive marketing solutions to our customers. We partner with brands to achieve meaningful progress, bringing value to our belief network through innovative, forward-thinking 360 solutions.",
            "values": "Integrity, Collaboration, Innovation, Transparency, Continuous Learning, Wellbeing, Client-Centric Approach, Adaptability, Team Collaboration, Work-Life Balance",
            "leadership": "Saliya Withange (CEO), Lahiru Halkewela (COO), Shezri Junaid (CSO), Rasanga Abhishek (CGO), Dhana Perera (Head of People and Culture), and others.",
            "policies": "Confidentiality: Do not disclose company info. Working Hours: 9 AM - 6 PM with flexibility. Leave: 7 days casual, 14 days annual (2nd year), 7 days medical (2nd year).",
            "fun_activities": "Houses: Sterling, Don, Peggy, Pete. Events: New Year Celebrations, Annual Trip, Sports Day, Secret Santa.",
            "skills_expertise": "Human resource management (HR), Culture building, Event planning, Employee welfare",
            "house": "Sterling",
            "team": json.dumps({
                "Saliya": {"position": "CEO", "skills": "Product marketing, Technology marketing, Marketing strategy, Public relations, Creative marketing", "sectors": "Fin Tech, SAAS, FMCG, Banking & finance", "house": "Don"},
                "Lahiru": {"position": "COO", "skills": "Project management, Web development, Technology marketing", "sectors": "SAAS, FMCG", "house": "Peggy"},
                "Shezri": {"position": "CSO", "skills": "Marketing strategy, Brand strategy, Social media marketing, Creative AI", "sectors": "FMCG, Tea", "house": "Sterling"},
                "Rasanga": {"position": "CGO", "skills": "Product development, AI, Marketing strategy, Data & analytics", "sectors": "Banking & Finance, SAAS, Fin Tech, Consulting", "house": "Pete"},
                "Faz": {"position": "POD 1 - Lead", "skills": "Digital marketing, Account based marketing, Marketing strategy, Project management, Technology marketing, LinkedIn Marketing", "sectors": "Pharmaceuticals, Health care, Beauty & personal care", "house": "Don"},
                "Azard": {"position": "POD 2 - Lead", "skills": "Digital marketing, Creative production, Outdoor promotions, ATL marketing, Channel strategy", "sectors": "Banking & finance, FMCG, Beauty & personal care, Consumer electronics, Retail", "house": "Don"},
                "Mel": {"position": "POD 3 - Lead", "skills": "Creative production, Channel strategy", "sectors": "Education, FMCG, Retail", "house": "Sterling"},
                "Dhana": {"position": "Head of HR", "skills": "Human resource management (HR), Culture building, Event planning, Employee welfare", "sectors": "", "house": "Sterling"},
                "Madushani": {"position": "Head of Finance", "skills": "Accounting, Payment management, Finance", "sectors": "", "house": "Don"}
            }),
            "top_clients": json.dumps({
                "MAS Holdings": {"country": "Global", "sector": "Clothing & textile"},
                "Hemas": {"country": "Sri Lanka", "sector": "Consumer goods"},
                "MSD": {"country": "Malaysia", "sector": "Pharmaceutical"},
                "Zuellig Pharma": {"country": "Malaysia", "sector": "Pharmaceutical"},
                "Singer": {"country": "Sri Lanka", "sector": "Retail"},
                "Keells": {"country": "Sri Lanka", "sector": "Retail"},
                "CBL": {"country": "Sri Lanka", "sector": "Food & beverage"},
                "Fortude": {"country": "Sri Lanka", "sector": "IT"},
                "Kaya": {"country": "USA", "sector": "IT"},
                "WSO2": {"country": "Global", "sector": "IT"},
                "TBWA": {"country": "Malaysia", "sector": "Advertising"}
            }),
            "benefits": "Insurance: LKR 1,000,000 cover, including maternity complications, dental, outpatient benefits for employee and spouse. Annual Trip: All-expense-paid company trip with one night out. Education Loans: Up to LKR 500,000 for selected courses.",
            "tools_software": json.dumps({
                "Google Business Suite": {"purpose": "Email, Calendar, Documents, Meetings", "access": "All employees"},
                "Similar Web": {"purpose": "Web analytics, Search analytics, SEO, Competitor analysis", "access": "Shezri"},
                "Timespot": {"purpose": "Leave management", "access": "Dhana"},
                "Figma": {"purpose": "Prototyping & design", "access": "Thareef"}
            })
        }
    }
    
    for label, metadata in knowledge_base.items():
        dummy_embedding = [0.0] * 128  # Placeholder embedding for Chroma
        collection.upsert(
            ids=[label],
            embeddings=[dummy_embedding],
            metadatas=[metadata]
        )
        print(f"Added {label} to ChromaDB with metadata: {metadata}")

# Uncomment to populate initially (run once), then comment out again
#populate_chroma_db()

# Set up upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define reference images
REFERENCE_IMAGES = {
    "saliya": "static/uploads/Saliya.jpg",
    "dhana": "static/uploads/Dhana-2.jpg"
}

def find_similar_image(uploaded_filepath):
    """Match uploaded image with reference images using DeepFace facial recognition."""
    print(f"Analyzing uploaded image: {uploaded_filepath}")
    
    min_distance = float('inf')
    matched_label = "unknown"
    matched_path = None
    
    for label, ref_path in REFERENCE_IMAGES.items():
        try:
            result = DeepFace.verify(
                img1_path=uploaded_filepath,
                img2_path=ref_path,
                model_name="VGG-Face",
                distance_metric="cosine",
                enforce_detection=True
            )
            distance = result["distance"]
            print(f"Distance to {label}: {distance}")
            if result["verified"] and distance < min_distance:
                min_distance = distance
                matched_label = label
                matched_path = f"/{ref_path}"
        except Exception as e:
            print(f"Error comparing with {label}: {e}")
    
    if matched_label == "unknown":
        print("No match found")
    else:
        print(f"Match found: {matched_label}")
    return matched_label, matched_path

def chatbot_response(label, metadata, user_input):
    """Generate chatbot response with session-based chat history, greeting only once."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Initialize session variables if not present
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'greeted' not in session:
        session['greeted'] = False
    if 'current_label' not in session:
        session['current_label'] = None
    
    chat_history = session['chat_history']
    history_str = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in chat_history])
    
    # Reset greeted flag if label changes
    if session['current_label'] != label:
        session['greeted'] = False
        session['current_label'] = label
    
    if label == "unknown":
        prompt = "The image couldn’t be recognized. Please tell the user to try another one in a friendly tone."
    else:
        info = metadata.get("info", "No info available.")
        company_context = metadata.get("who_we_are", "No company info available.")
        if not session['greeted'] and not user_input:  # Initial greeting
            prompt = f"Respond as if you are {label.capitalize()} from this info: '{info}'. Introduce yourself in the first person with a friendly tone, mentioning this shared company context: '{company_context}'. Ask how you can assist today."
            session['greeted'] = True
        elif user_input:  # Follow-up response
            prompt = f"Respond as if you are {label.capitalize()} from this info: '{info}'. Answer the user’s question '{user_input}' naturally in the first person, like a human would, using this company context if relevant: '{company_context}'. Additional details you can use if needed: Values: '{metadata.get('values')}', Leadership: '{metadata.get('leadership')}', Policies: '{metadata.get('policies')}', Fun Activities: '{metadata.get('fun_activities')}'. Previous chat history:\n{history_str}"
        else:  # No input after greeting
            prompt = f"Respond as if you are {label.capitalize()} from this info: '{info}'. Say something brief and friendly to keep the conversation going, using this company context if relevant: '{company_context}'. Previous chat history:\n{history_str}"
    
    try:
        response = model.generate_content(prompt)
        if user_input:
            chat_history.append({"sender": "You", "text": user_input})
        chat_history.append({"sender": label.capitalize(), "text": response.text})
        session['chat_history'] = chat_history[-10:]  # Limit to last 10 messages
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I couldn’t generate a response right now."

@app.route('/')
def index():
    return "Welcome to Enfection Chat API"

@app.route('/upload', methods=['POST'])
def upload_image():
    print("Received upload request")
    if 'image' not in request.files:
        print("No image in request")
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        print("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_image.jpg")
    file.save(filepath)
    print(f"Image saved to {filepath}")
    
    # Reset chat history and greeted status for new upload
    session['chat_history'] = []
    session['greeted'] = False
    
    label, ref_image_path = find_similar_image(filepath)
    print(f"Found label: {label}")
    
    if label != "unknown":
        result = collection.get(ids=[label], include=["metadatas"])
        metadata = result["metadatas"][0] if result["metadatas"] else {}
        print(f"Metadata retrieved for {label}: {metadata}")
    else:
        metadata = {}
        print("No metadata for unknown label")
    
    response = chatbot_response(label, metadata, None)
    
    return jsonify({
        "image_path": "/static/uploads/uploaded_image.jpg",
        "reference_image_path": ref_image_path,
        "response": response,
        "label": label
    })

@app.route('/chat', methods=['POST'])
def chat():
    print("Received chat request")
    data = request.get_json()
    if not data:
        print("No JSON data received")
        return jsonify({"error": "No data provided"}), 400
    
    label = data.get("label", "unknown")
    user_input = data.get("message", "")
    print(f"Chat input: {user_input}, label: {label}")
    
    if label != "unknown":
        result = collection.get(ids=[label], include=["metadatas"])
        metadata = result["metadatas"][0] if result["metadatas"] else {}
        print(f"Metadata retrieved for {label}: {metadata}")
    else:
        metadata = {}
        print("No metadata for unknown label")
    
    response = chatbot_response(label, metadata, user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    print("Starting Flask app")
    app.run(debug=True, use_reloader=False, port=5000)