import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import speech_recognition as sr
import requests
import json
from PIL import Image
import pytesseract
import asyncio
import base64
import logging
import re
from typing import Optional, Tuple
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from typing import List
# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from meeting_report_emailer import MeetingReportEmailer
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
#CARD_STREAM_URL = os.getenv('CARD_STREAM_URL', 'rtsp://metaverse911:hellomoto123@192.168.1.106:554/stream1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PDF_FOLDER = os.getenv('PDF_FOLDER', 'data/datamn')
if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found in environment variables!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str
# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize LangChain components
qa_chain = None
try:
    # Load PDF documents
    loaders = [PyPDFLoader(os.path.join(PDF_FOLDER, fn)) 
              for fn in os.listdir(PDF_FOLDER) if fn.endswith('.pdf')]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    logger.info("LangChain QA chain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize QA chain: {str(e)}")
    qa_chain = None


from langchain.prompts import PromptTemplate

# Custom prompt template
template = """You are a helpful AI assistant with access to knowledge about UBIK and subsidiaries. 
Answer questions based on the provided context. If you don't know something or if it's not in the context, 
say that you aren't trained for it. Also, don't go out of context. Your name is UBIK AI. Answer mostly under 50 words unless very much required.

Context: {context}
Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Create QA chain with source documents tracking
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)
# Track detection states
face_detected = False
card_detected = False


# Card stream URL from environment variable


class MeetingTracker:
  
    def __init__(self, storage_dir: str = "meeting_logs"):
        self.storage_dir = storage_dir
        self.current_meeting = {
            "start_time": None,
            "end_time": None,
            "participant_name": None,
            "participant_email": "NA",
            "participant_phone": "NA",
            "participant_company": "NA",
            "questions": [],
            "responses": [],
            "discussion_overview": None
        }
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    async def generate_discussion_overview(self) -> str:
        """Generate a discussion overview using GPT API"""
        if not self.current_meeting["questions"]:
            return "No discussion took place."

        # Prepare conversation history for GPT
        conversation = ""
        for q, r in zip(self.current_meeting["questions"], self.current_meeting["responses"]):
            conversation += f"Q: {q}\nA: {r}\n\n"

        prompt = (
            "Based on the following conversation, provide a concise overview of what was discussed. "
            "Focus on the main points, key decisions, and important information shared. "
            "Keep it to 2-3 sentences.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {OPENAI_API_KEY}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'gpt-3.5-turbo',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0.3
                    }
                )
            )
            
            overview = response.json()['choices'][0]['message']['content'].strip()
            self.current_meeting["discussion_overview"] = overview
            return overview
        except Exception as e:
            logger.error(f"Error generating discussion overview: {str(e)}")
            return "Error generating discussion overview."

    def start_meeting(self, participant_name: Optional[str] = None):
        """Start a new meeting session"""
        self.current_meeting["start_time"] = datetime.now().isoformat()
        self.current_meeting["participant_name"] = participant_name
        self.current_meeting["questions"] = []
        self.current_meeting["responses"] = []
        self.current_meeting["topics_discussed"] = set()

    def update_contact_info(self, email: str, phone: str, company: str):
        """Update participant contact information"""
        self.current_meeting["participant_email"] = email
        self.current_meeting["participant_phone"] = phone
        self.current_meeting["participant_company"] = company

    def _save_meeting_log(self, summary: Dict) -> str:
        """Save meeting summary to a file with contact information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        participant = self.current_meeting["participant_name"] or "anonymous"
        filename = f"{self.storage_dir}/meeting_{participant}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Meeting Summary ===\n\n")
            f.write(f"Date: {summary['meeting_date']}\n")
            f.write(f"Time: {summary['start_time']} - {summary['end_time']}\n")
            f.write(f"Duration: {summary['duration_minutes']:.1f} minutes\n")
            f.write(f"Participant: {summary['participant_name']}\n")
            f.write(f"Email: {self.current_meeting['participant_email']}\n")
            f.write(f"Phone: {self.current_meeting['participant_phone']}\n")
            f.write(f"Company: {self.current_meeting['participant_company']}\n")
            f.write(f"Total Questions Asked: {summary['total_questions']}\n\n")
            
            f.write("=== Discussion Overview ===\n")
            f.write(f"{summary['discussion_overview']}\n\n")
            
            f.write("=== Questions and Responses ===\n\n")
            for i, qa in enumerate(summary['questions_and_responses'], 1):
                f.write(f"Q{i}: {qa['Q']}\n")
                f.write(f"A{i}: {qa['A']}\n\n")
        
        return filename

    def add_interaction(self, question: str, response: str):
        """Record a Q&A interaction"""
        self.current_meeting["questions"].append(question)
        self.current_meeting["responses"].append(response)
        
        # Extract potential topics from question and response
        # This is a simple implementation - could be enhanced with NLP
        words = set((question + " " + response).lower().split())
        self.current_meeting["topics_discussed"].update(words)

    async def end_meeting(self) -> str:
        """End the meeting and save the summary"""
        if not self.current_meeting["start_time"]:
            return "No active meeting to end"

        self.current_meeting["end_time"] = datetime.now().isoformat()
        
        # Generate discussion overview
        await self.generate_discussion_overview()
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save to file
        filename = self._save_meeting_log(summary)
        
        # Reset current meeting
        self.current_meeting = {
            "start_time": None,
            "end_time": None,
            "participant_name": None,
            "participant_email": "NA",
            "participant_phone": "NA",
            "participant_company": "NA",
            "questions": [],
            "responses": [],
            "discussion_overview": None
        }
        
        return filename

    async def generate_discussion_overview(self) -> str:
        """Generate a discussion overview using GPT API"""
        if not self.current_meeting["questions"]:
            return "No discussion took place."

        # Prepare conversation history for GPT
        conversation = ""
        for q, r in zip(self.current_meeting["questions"], self.current_meeting["responses"]):
            conversation += f"Q: {q}\nA: {r}\n\n"

        prompt = (
            "Based on the following conversation, provide a concise overview of what was discussed. "
            "Focus on the main points, key decisions, and important information shared. "
            "Keep it to 2-3 sentences.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {OPENAI_API_KEY}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'gpt-3.5-turbo',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0.3
                    }
                )
            )
            
            overview = response.json()['choices'][0]['message']['content'].strip()
            self.current_meeting["discussion_overview"] = overview
            return overview
        except Exception as e:
            logger.error(f"Error generating discussion overview: {str(e)}")
            return "Error generating discussion overview."
            
    def _generate_summary(self) -> Dict:
        """Generate a meeting summary"""
        start_time = datetime.fromisoformat(self.current_meeting["start_time"])
        end_time = datetime.fromisoformat(self.current_meeting["end_time"])
        duration = end_time - start_time
        
        summary = {
            "meeting_date": start_time.strftime("%Y-%m-%d"),
            "start_time": start_time.strftime("%H:%M:%S"),
            "end_time": end_time.strftime("%H:%M:%S"),
            "duration_minutes": duration.total_seconds() / 60,
            "participant_name": self.current_meeting["participant_name"],
            "total_questions": len(self.current_meeting["questions"]),
            "questions_and_responses": [
                {"Q": q, "A": r} for q, r in zip(
                    self.current_meeting["questions"],
                    self.current_meeting["responses"]
                )
            ],
            "discussion_overview": self.current_meeting["discussion_overview"]
        }
        
        return summary
    
FACE_STREAM_URL = os.getenv('FACE_STREAM_URL', 'rtsp://metaverse911keshav:hellomoto@123@192.168.1.111:554/stream1')

async def init_face_stream():
    """Initialize RTSP stream for face detection"""
    try:
        logger.info(f"Attempting to connect to face RTSP stream at: {FACE_STREAM_URL}")
        
        # Try to create VideoCapture object
        stream = cv2.VideoCapture(FACE_STREAM_URL, cv2.CAP_FFMPEG)
        
        # Log initial state
        logger.info(f"Initial stream open status: {stream.isOpened()}")
        
        # Get and log stream properties
        fps = stream.get(cv2.CAP_PROP_FPS)
        frame_width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Stream properties - FPS: {fps}, Width: {frame_width}, Height: {frame_height}")
        
        # Set buffer size
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test read a frame
        ret, frame = stream.read()
        if not ret:
            logger.error("Failed to read first frame from stream")
            return None
        
        if not stream.isOpened():
            logger.error(f"Failed to open RTSP stream: {FACE_STREAM_URL}")
            return None
            
        logger.info("Successfully connected to RTSP stream and read first frame")
        return stream
    except Exception as e:
        logger.error(f"Error initializing RTSP stream: {str(e)}")
        return None

async def check_face_stream(face_cascade):
    """Check RTSP stream for faces, returns True if face detected"""
    logger.info("Starting face detection from stream")
    
    face_stream = await init_face_stream()
    if not face_stream:
        logger.error("Failed to initialize face stream")
        raise Exception("Unable to access face stream")

    start_time = asyncio.get_event_loop().time()
    timeout = 100000000000000000000000000
    frame_count = 0

    try:
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            logger.debug("Attempting to read frame from face stream")
            ret, frame = await asyncio.get_event_loop().run_in_executor(
                None, face_stream.read
            )
            
            if not ret:
                logger.warning("Failed to read frame from face stream")
                await asyncio.sleep(0.1)
                continue

            frame_count += 1
            logger.info(f"Processing frame {frame_count} for face detection")

            try:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    logger.info(f"Face detected in frame {frame_count}")
                    return True
                    
            except Exception as e:
                logger.error(f"Face detection error on frame {frame_count}: {str(e)}")
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Face detection timed out after {frame_count} frames")
        return False
    finally:
        logger.info("Releasing face stream")
        face_stream.release()



async def get_ai_response(text: str, name: Optional[str] = None) -> str:
    """Get response using PDF content and extract contact information"""
    if not qa_chain:
        return "System is not ready. Please try again later."
    
    try:
        # First, extract contact information using GPT
        contact_prompt = (
            "Extract email, phone number, and company name from the following text. "
            "Respond in JSON format with keys 'email', 'phone', 'company'. "
            "Use 'NA' if information is not found.\n\n"
            f"Text: {text}"
        )
        
        contact_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': contact_prompt}],
                    'temperature': 0.1
                }
            )
        )
        
        contact_info = contact_response.json()['choices'][0]['message']['content']
        contact_data = json.loads(contact_info)
        
        # Store contact information
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        contact_filename = f"contact_logs/contact_{timestamp}.txt"
        
        # Ensure directory exists
        os.makedirs("contact_logs", exist_ok=True)
        
        # Save contact information
        with open(contact_filename, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Name: {name if name else 'NA'}\n")
            f.write(f"Email: {contact_data.get('email', 'NA')}\n")
            f.write(f"Phone: {contact_data.get('phone', 'NA')}\n")
            f.write(f"Company: {contact_data.get('company', 'NA')}\n")
            f.write(f"Original Text: {text}\n")

        # Get response from QA chain
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_chain({"query": text})
        )
        
        if not result.get('source_documents'):
            return "I can only answer questions about Metaverse 911 and its services from the provided documents."
            
        response = result['result'].strip()
        
        if name:
            response = f"{name}, {response}"
            
        return response
        
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return "Error processing your question. Please try again."
    
face_detected = False
card_detected = False

# Card stream URL from environment variable
CARD_STREAM_URL = os.getenv('CARD_STREAM_URL', 'rtsp://metaverse911:hellomoto123@192.168.1.106:554/stream1')


async def validate_contact_info(text: str) -> Tuple[bool, Optional[Dict]]:
    """
    Validate if the text contains a valid name and extract contact information using AI.
    Returns (is_valid, contact_info_dict)
    """
    logger.info(f"Attempting to validate text and extract contact info: {text!r}")
    
    try:
        # Prepare the prompt for the AI to extract all contact information
        prompt = (
            "Extract name and contact information from the text. If a piece of information "
            "is not found, use 'NA'. Respond in this exact JSON format:\n"
            "{\n"
            "  'is_valid': true/false,\n"
            "  'name': 'extracted name or NA',\n"
            "  'email': 'email or NA',\n"
            "  'phone': 'phone number or NA',\n"
            "  'company': 'company name or NA'\n"
            "}\n\n"
            f"Text to analyze: {text}"
        )
        
        logger.debug(f"Sending prompt to OpenAI: {prompt}")
        
        # Call OpenAI API
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1
                },
                timeout=5
            )
        )
        response.raise_for_status()
        
        # Extract AI's response
        ai_response = response.json()['choices'][0]['message']['content'].strip()
        logger.info(f"AI response: {ai_response!r}")
        
        # Parse the JSON response
        contact_info = json.loads(ai_response)
        
        # Log the extracted information
        logger.info(f"Extracted contact info: {contact_info}")
        
        return contact_info['is_valid'], contact_info
        
    except Exception as e:
        logger.error(f"Error in AI contact info validation: {str(e)}")
        return False, None

# Modified check_card_stream function to use the new validate_contact_info
async def check_card_stream():
    """Check RTSP stream for card with valid contact info, returns contact info dict or None"""
    logger.info("Starting card stream check")
    
    card_stream = await init_card_stream()
    if not card_stream:
        logger.error("Failed to initialize card stream")
        raise Exception("Unable to access card stream")

    start_time = asyncio.get_event_loop().time()
    timeout = 3  # seconds
    frame_count = 0

    try:
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            logger.debug("Attempting to read frame from card stream")
            ret, frame = await asyncio.get_event_loop().run_in_executor(
                None, card_stream.read
            )
            
            if not ret:
                logger.warning("Failed to read frame from card stream")
                await asyncio.sleep(1)
                continue

            frame_count += 1
            logger.info(f"Successfully read frame {frame_count} from card stream")

            try:
                # Convert frame to bytes
                success, encoded_frame = cv2.imencode('.jpg', frame)
                if not success:
                    continue
                
                # Process frame with Vision API
                text = await process_image_with_vision_api(encoded_frame.tobytes())
                logger.info(f"Card stream Vision API text from frame {frame_count}: {text!r}")
                
                if text:
                    is_valid, contact_info = await validate_contact_info(text)
                    if is_valid:
                        logger.info(f"Valid contact info found in frame {frame_count}: {contact_info}")
                        return contact_info
            except Exception as e:
                logger.error(f"Card processing error on frame {frame_count}: {str(e)}")
            
            await asyncio.sleep(1)
        
        logger.warning(f"Card stream check timed out after {frame_count} frames")
        return None
    finally:
        logger.info("Releasing card stream")
        card_stream.release()




GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY')


async def process_image_with_vision_api(image_bytes: bytes) -> str:
    """
    Process image using Google Cloud Vision API
    """
    try:
        # Convert image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare the request payload
        vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_b64
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }

        # Make request to Vision API
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                vision_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
        )
        response.raise_for_status()

        # Extract text from response
        result = response.json()
        if 'responses' in result and result['responses']:
            text_annotation = result['responses'][0].get('textAnnotations', [])
            if text_annotation:
                return text_annotation[0].get('description', '')
        
        return ''

    except Exception as e:
        logger.error(f"Error in Vision API processing: {str(e)}")
        return ''


async def init_card_stream():
    try:
        logger.info(f"Attempting to connect to RTSP stream at: {CARD_STREAM_URL}")
        
        # Try to create VideoCapture object
        stream = cv2.VideoCapture(CARD_STREAM_URL, cv2.CAP_FFMPEG)
        
        # Log initial state
        logger.info(f"Initial stream open status: {stream.isOpened()}")
        
        # Get and log stream properties
        fps = stream.get(cv2.CAP_PROP_FPS)
        frame_width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Stream properties - FPS: {fps}, Width: {frame_width}, Height: {frame_height}")
        
        # Set buffer size
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test read a frame
        ret, frame = stream.read()
        if not ret:
            logger.error("Failed to read first frame from stream")
            return None
        
        if not stream.isOpened():
            logger.error(f"Failed to open RTSP stream: {CARD_STREAM_URL}")
            return None
            
        logger.info("Successfully connected to RTSP stream and read first frame")
        return stream
    except cv2.error as e:
        logger.error(f"OpenCV error initializing stream: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error initializing RTSP stream: {str(e)}")
        logger.exception("Full exception traceback:")
        return None
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Initialize state variables
    current_contact_info = None
    face_detected = False
    card_task = None
    face_task = None
    
    # Initialize meeting tracker and emailer
    meeting_tracker = MeetingTracker()
    emailer = MeetingReportEmailer()
    meeting_tracker.start_meeting()
    logger.info("Meeting tracking started")
    
    try:
        # Start face detection task immediately
        face_task = asyncio.create_task(check_face_stream(face_cascade))
        
        while True:
            # Check for WebSocket messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                
                if data.startswith('question:'):
                    question = data[9:]
                    
                    if not face_detected:
                        response = "Please wait while we detect your face."
                    elif card_task and not card_task.done():
                        response = "Please wait while we verify your card."
                    else:
                        try:
                            response = await get_ai_response(question, current_contact_info['name'] if current_contact_info else None)
                            meeting_tracker.add_interaction(question, response)
                        except Exception as e:
                            logger.error(f"Error getting AI response: {str(e)}")
                            response = "I apologize, but I encountered an error processing your question. Please try again."
                    
                    await websocket.send_text(f"ai_response:{response}")
                
                elif data == "end_session":
                    logger.info("Session end requested by client")
                    break
            
            except asyncio.TimeoutError:
                pass
            
            # Check face detection task
            if face_task and face_task.done():
                try:
                    face_detected = face_task.result()
                    if face_detected:
                        await websocket.send_text("face_detected")
                        logger.info("Face detected, starting card detection")
                        # Start card detection
                        card_task = asyncio.create_task(check_card_stream())
                    face_task = None
                except Exception as e:
                    logger.error(f"Face detection failed: {str(e)}")
                    face_task = None
            
            # Check card task completion
            if card_task and card_task.done():
                try:
                    current_contact_info = card_task.result()
                    if current_contact_info:
                        # Update meeting tracker with contact information
                        meeting_tracker.current_meeting["participant_name"] = current_contact_info['name']
                        meeting_tracker.update_contact_info(
                            email=current_contact_info['email'],
                            phone=current_contact_info['phone'],
                            company=current_contact_info['company']
                        )
                        logger.info(f"Card detected, participant info: {current_contact_info}")
                        
                        await websocket.send_text(f"card_detected:{json.dumps(current_contact_info)}")
                        welcome_message = f"Welcome {current_contact_info['name']}! How can I assist you today?"
                        await websocket.send_text(f"ai_response:{welcome_message}")
                    else:
                        logger.warning("No card detected")
                        await websocket.send_text("no_card_detected")
                        await websocket.send_text("ai_response:No card found. You can still proceed to ask questions.")
                    card_task = None
                except Exception as e:
                    logger.error(f"Card detection failed: {str(e)}")
                    await websocket.send_text("card_error:Card verification failed")
                    await websocket.send_text("ai_response:There was an error reading your card. Please try again.")
                    card_task = None
                    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up tasks
        if face_task and not face_task.done():
            face_task.cancel()
        if card_task and not card_task.done():
            card_task.cancel()
            
        # End meeting and save log
        try:
            log_filename = await meeting_tracker.end_meeting()
            logger.info(f"Meeting log saved to: {log_filename}")
            emailer.process_and_send_report(log_filename)

            
            # Send email if we have a valid email address
            participant_email = meeting_tracker.current_meeting.get("participant_email")
            if participant_email and participant_email != "NA":
                try:
                    base_filename = os.path.basename(log_filename)
                    success = emailer.process_and_send_report(base_filename)
                    if success:
                        logger.info(f"Meeting summary email sent to {participant_email}")
                    else:
                        logger.error("Failed to send meeting summary email")
                except Exception as e:
                    logger.error(f"Error sending meeting summary email: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error saving meeting log: {str(e)}")
        
        await websocket.close()
        logger.info("WebSocket connection closed")
def estimate_speech_duration(text: str) -> float:
    # Average speaking rate is about 150 words per minute
    # which means about 2.5 words per second
    
    # Clean the text of punctuation and split into words
    words = re.findall(r'\w+', text.lower())
    word_count = len(words)
    
    # Calculate estimated duration in seconds
    # Adding a small buffer for natural pauses
    estimated_seconds = (word_count / 2.5) + 1
    
    # Add additional time for longer sentences
    sentences = len(re.findall(r'[.!?]+', text)) or 1
    pause_time = sentences * 0.5  # Add 0.5 seconds per sentence for natural pauses
    
    return estimated_seconds + pause_time

@app.post("/estimate-duration")
async def get_speech_duration(request: TextRequest) -> dict:
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    duration = estimate_speech_duration(request.text)
    return {"estimated_duration": duration}
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)