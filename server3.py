import os
import uuid
import time
import logging
import threading
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
import speech_recognition as sr
import os
import uuid
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import subprocess
import logging
import random
from typing import Dict, List
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')

# Default system message
DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant with access to knowledge about UBIK solutions and subsidaries. 
with my prompt u will also receive the context regarding the question, answer based on that, if question isnt based on the context then say you are not trained to answer it , in a polite manner. you will answer the question that user will ask. you can respond to greetings or general greet things, Also, don't go out of context. Your name is UBIK AI. Answer mostly under 50 words unless very much required.
there will be spelling errors, please consider them and give best response possible"""

logging.basicConfig(
    filename='app_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=3)

global_vectorstore = None
vectorstore_lock = threading.Lock()

user_sessions = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRODUCT_RESPONSES: Dict[str, List[str]] = {
    "ethiglo": [
        "Ethiglo is one of our premier skincare solutions",
        "Let me tell you about our advanced Ethiglo formulation",
        "Ethiglo represents our commitment to quality skincare",
        "Our Ethiglo product line has been making waves",
        "Speaking about Ethiglo's remarkable benefits"
    ],
    "derma": [
        "Our Derma range is specially formulated",
        "The Derma line represents cutting-edge skincare",
        "Our Derma products are trusted by professionals",
        "Looking at our innovative Derma range",
        "Regarding our specialized Derma products"
    ]
    # Add more products as needed
}

# Emotion-based responses
EMOTIONAL_RESPONSES = {
    "urgent": [
        "I understand this is urgent - let me explain about",
        "Let me quickly tell you about",
        "I'll immediately share the details about",
        "Here's the urgent information about",
        "Let me address your urgent query about"
    ],
    "frustrated": [
        "I understand your concern - let me clarify about",
        "I appreciate your patience - let me explain about",
        "Let me help resolve your concerns about",
        "I'll help you understand everything about",
        "Let me address your concerns regarding"
    ],
    "curious": [
        "That's an interesting question about",
        "I'm excited to share information about",
        "Let me explain the fascinating details of",
        "You'll be interested to know about",
        "Let me share what makes this special -"
    ],
    "technical": [
        "Looking at the technical aspects,",
        "From a technical perspective,",
        "Analyzing the specifications,",
        "Examining the technical details,",
        "Regarding the technical features,"
    ],
    "comparison": [
        "When comparing the options,",
        "Looking at the differences,",
        "Analyzing the various choices,",
        "Examining the alternatives,",
        "Comparing the features,"
    ],
    "price": [
        "Regarding the pricing structure,",
        "Looking at the cost aspects,",
        "About the pricing details,",
        "Concerning the investment value,",
        "Speaking about the pricing,"
    ]
}

def log_event(event_type: str, details: str = "", user_id: str = "Unknown"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} | {event_type} | UserID: {user_id} | {details}\n"
    with open("app_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_line)

def get_user_state(user_id: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'memory': ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            ),
            'chain': None,
            'history': [],
            'system_message': DEFAULT_SYSTEM_MESSAGE
        }
        log_event("UserJoined", "New user state created.", user_id=user_id)
    return user_sessions[user_id]

@lru_cache(maxsize=128)
def get_pdf_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return ""

def get_text_chunks(text: str):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

import os
import uuid
import json
import base64
import aiohttp
import logging
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def handle_speech_to_text(file: UploadFile):
    logger.info(f"Starting speech to text conversion for file: {file.filename}")
    
    if file is None:
        logger.error("No file provided")
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided."}
        )

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate temporary file paths
    webm_path = os.path.join("data", f"temp_{uuid.uuid4().hex}.webm")
    wav_path = os.path.join("data", f"temp_{uuid.uuid4().hex}.wav")
    
    try:
        # Save the uploaded file
        logger.debug("Reading uploaded file")
        content = await file.read()
        with open(webm_path, "wb") as f:
            f.write(content)
        logger.debug(f"Saved WebM file to {webm_path}")
        
        # Convert to WAV format
        logger.debug("Converting WebM to WAV")
        try:
            subprocess.run([
                'ffmpeg', '-i', webm_path, 
                '-acodec', 'pcm_s16le', 
                '-ar', '16000', 
                '-ac', '1', 
                wav_path
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Audio conversion failed: {e.stderr}"}
            )

        # Initialize speech recognizer
        r = sr.Recognizer()
        
        # Process the audio file
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)  # Read the entire audio file
            logger.debug("Processing audio with Google Speech Recognition...")
            
            try:
                text = r.recognize_google(audio)
                logger.info("Successfully transcribed audio")
                return {"status": "success", "text": text}
                
            except sr.UnknownValueError:
                logger.error("Google Speech Recognition could not understand audio")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not understand audio"}
                )
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google SR service; {e}")
                return JSONResponse(
                    status_code=503,
                    content={"error": f"Speech service error: {str(e)}"}
                )

    except Exception as e:
        logger.exception("Unexpected error occurred")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during transcription: {str(e)}"}
        )
    
    finally:
        # Clean up temporary files
        logger.debug("Cleaning up temporary files")
        for path in [webm_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Removed {path}")

def initialize_global_vectorstore():
    global global_vectorstore
    with vectorstore_lock:
        if global_vectorstore is not None:
            return True, "[SYSTEM MESSAGE] Vectorstore is already initialized."

        pdf_paths = [
            os.path.join('data', "Ilesh Sir (IK) - Words.pdf"),
            os.path.join('data', "UBIK SOLUTION.pdf"),
            os.path.join('data', "illesh3.pdf"),
            os.path.join('data', "website-data-ik.pdf")
        ]

        combined_text = ""
        for path in pdf_paths:
            combined_text += get_pdf_text(path) + " "

        if not combined_text.strip():
            return False, "No text could be extracted from the PDFs."

        text_chunks = get_text_chunks(combined_text)
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        global_vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        logger.info("Vectorstore has been created with OpenAI embeddings.")
    return True, "[SYSTEM MESSAGE] Vectorstore was created successfully."
def detect_emotion(question: str) -> str:
    question_lower = question.lower()
    
    # Emotion detection patterns
    patterns = {
        'urgent': ['urgent', 'asap', 'emergency', 'quick', 'fast', 'now'],
        'frustrated': ['not working', 'problem', 'issue', 'wrong', 'bad', 'help', 'confused'],
        'curious': ['why', 'how', 'what', 'curious', 'interested', 'tell me about', 'explain'],
        'technical': ['spec', 'technical', 'configuration', 'setup', 'system', 'work', 'function'],
        'comparison': ['compare', 'difference', 'better', 'versus', 'vs', 'or', 'which'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'discount', 'offer', 'worth']
    }
    
    for emotion, keywords in patterns.items():
        if any(word in question_lower for word in keywords):
            return emotion
    
    return "curious"  # Default emotion

def get_bridge_response(question: str) -> str:
    question_lower = question.lower()
    
    # First check for specific products
    for product, responses in PRODUCT_RESPONSES.items():
        if product in question_lower:
            return random.choice(responses)
    
    # If no product match, use emotion-based response
    emotion = detect_emotion(question)
    return random.choice(EMOTIONAL_RESPONSES[emotion])

def handle_userinput(user_question: str, user_id: str):
    user_state = get_user_state(user_id)
    conversation_chain = user_state['chain']
    if not conversation_chain:
        return None

    # Get bridge response
    bridge_response = get_bridge_response(user_question)
    
    # Create a prompt that instructs the AI to continue from the bridge response
    modified_question = f"""
    Continue this response seamlessly, providing accurate and detailed information:
    "{bridge_response}"
    
    Original question: {user_question}
    
    Keep the tone consistent and expand on the information naturally.
    """
    
    # Get AI response
    ai_response = conversation_chain({'question': modified_question})['answer'].strip()
    
    # Combine responses seamlessly
    final_response = f"{bridge_response} - {ai_response}"

    # Update history and logs
    user_state['history'].append((user_question, final_response))
    log_event("UserQuestion", f"Q: {user_question}", user_id=user_id)
    log_event("AIAnswer", f"A: {final_response}", user_id=user_id)

    return {'text': final_response}

def create_or_refresh_user_chain(user_id: str):
    user_state = get_user_state(user_id)
    if user_state['chain'] is None:
        if global_vectorstore is None:
            return False, "Global vectorstore is not initialized."

        # Create chat model with system message
        chat_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.9)

        # Create custom QA prompt template that includes system message
        condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

        qa_template = f"""
        {user_state['system_message']}

        Context: {{context}}
        
        Question: {{question}}
        
        Answer: """
        
        QA_PROMPT = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )

        # Create the chain
        user_state['chain'] = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=global_vectorstore.as_retriever(),
            memory=user_state['memory'],
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT}
        )
        
        logger.info(f"New conversation chain created for user {user_id} with system message: {user_state['system_message']}")
        return True, "Conversation chain created."
    else:
        return True, "Conversation chain already exists."


@app.on_event("startup")
async def startup_event():
    success, message = initialize_global_vectorstore()
    if not success:
        logger.error(f"Failed to initialize vectorstore: {message}")
    else:
        logger.info("Vectorstore initialized successfully on startup")

@app.get("/")
async def hello_root():
    return {
        "message": "Hello from FastAPI server. Use the endpoints to process data or ask questions.",
        "status": "operational"
    }

@app.post("/refresh_chain")
async def refresh_chain(request: Request):
    data = await request.json()
    if "user_id" not in data:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user_id."}
        )

    user_id = data["user_id"]
    success, message = create_or_refresh_user_chain(user_id)
    status = 'success' if success else 'error'
    return {"status": status, "message": message}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload."})

    if "user_id" not in data or "question" not in data:
        return JSONResponse(status_code=400, content={"error": "Missing user_id or question."})

    user_id = data["user_id"]
    user_question = data["question"]
    FORWARD_ENDPOINT = os.getenv("FORWARD_ENDPOINT", "https://win-ldm7g6fmsik.tail5e2acc.ts.net/receive")

    if user_id not in user_sessions or user_sessions[user_id]['chain'] is None:
        create_or_refresh_user_chain(user_id)

    answer = handle_userinput(user_question, user_id)
    if not answer:
        return {"status": "error", "message": "No conversation chain or unable to handle question."}

    # Forward the response to the endpoint
    try:
        # Format payload to match ReceiveText model
        payload = {
            "text": answer['text']  # Just send the text field as expected by ReceiveText
        }
        
        forward_response = requests.post(FORWARD_ENDPOINT, json=payload)
        forwarding_status = "success" if forward_response.status_code == 200 else "failed"
        
        if forward_response.status_code != 200:
            logger.error(f"Forward request failed with status {forward_response.status_code}: {forward_response.text}")
        else:
            logger.info(f"Response forwarding {forwarding_status}")
            
    except Exception as e:
        logger.error(f"Error forwarding response: {e}")
        forwarding_status = "failed"

    prompt_sent = {'question': user_question}
    return {
        "status": "success", 
        "data": answer, 
        "prompt": prompt_sent,
        "forwarding_status": forwarding_status
    }

@app.post("/set_system_message")
async def set_system_message(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id")
    system_message = data.get("system_message")

    if not user_id or not system_message:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing user_id or system_message."}
        )

    user_state = get_user_state(user_id)
    user_state['system_message'] = system_message
    # Force recreation of chain with new system message
    user_state['chain'] = None
    
    success, message = create_or_refresh_user_chain(user_id)
    
    log_event(
        "SystemMessageUpdated",
        f"System message updated to: {system_message}",
        user_id=user_id
    )
    
    return {
        "status": "success" if success else "error",
        "message": f"System message updated. {message}"
    }

@app.get("/get_system_message")
async def get_system_message(user_id: str):
    if not user_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user_id."}
        )

    user_state = get_user_state(user_id)
    return {
        "status": "success",
        "system_message": user_state['system_message']
    }

@app.post("/clear_history")
async def clear_history(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id", None)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id."})

    if user_id in user_sessions:
        user_state = user_sessions[user_id]
        user_state['memory'].clear()
        user_state['history'].clear()
        user_state['chain'] = None
        log_event("HistoryCleared", "User cleared their chat history.", user_id=user_id)
        return {"status": "success", "message": "Chat history cleared."}
    else:
        return {"status": "error", "message": "No user session found to clear."}

@app.post("/speech_to_text")
async def speech_to_text_endpoint(file: UploadFile = File(...)):
    return await handle_speech_to_text(file)

@app.post("/logout")
async def logout(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id", None)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id."})

    if user_id in user_sessions:
        del user_sessions[user_id]
    return {"status": "success", "message": "User session cleared."}


# ------------------------------------------------------------------------
#       GOOGLE CLOUD TEXT-TO-SPEECH ENDPOINTS USING API KEY
# ------------------------------------------------------------------------

GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")  # Load your API key from .env or environment variable

@app.post("/text_to_speech")
async def text_to_speech_api(request: Request):
    """
    Convert text to speech using Google Cloud TTS with API key authorization.
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        voice_name = data.get("voice", "en-US-Wavenet-D")  # Default to a US English voice
        language_code = data.get("language_code", "en-US")
        speaking_rate = data.get("speaking_rate", 1.0)  # Normal speed

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'text' in request."}
            )

        # Google TTS API endpoint with API key
        url = "https://texttospeech.googleapis.com/v1/text:synthesize?key=" + GOOGLE_TTS_API_KEY

        # Construct the payload
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": speaking_rate,
            },
        }

        # Make request to the Google TTS API
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            audio_content = response.json().get("audioContent", None)
            if not audio_content:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No audio content received from TTS API."}
                )

            # Decode the base64 audio content and save it
            audio_path = os.path.join("data", f"tts_output_{uuid.uuid4().hex}.mp3")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(base64.b64decode(audio_content))

            # Return path or URL to the generated audio
            return {"status": "success", "audio_url": audio_path}
        else:
            # If there's an error from Google
            return JSONResponse(
                status_code=response.status_code,
                content={"error": response.text}
            )

    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during text-to-speech synthesis."}
        )


# You can remove or comment out the StaticFiles mounting since we're not storing files anymore
# app.mount("/data", StaticFiles(directory="data"), name="data")
#pip install sentence-transformers transformers torch
# Command to run:
# uvicorn server3:app --host 0.0.0.0 --port 8000