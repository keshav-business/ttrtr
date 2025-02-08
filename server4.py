from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import os
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
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
import speech_recognition as sr
import base64
import io
from fastapi import FastAPI, File, UploadFile, Form
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import os
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage class
class QuizStorage:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.user_sessions: Dict[str, Dict] = {}

quiz_storage = QuizStorage()

# Pydantic models
class QuizResponse(BaseModel):
    question_id: int
    answer: str

class VoiceInput(BaseModel):
    audio_data: str
    user_id: str

class QuizResult(BaseModel):
    total_score: float
    max_score: float
    feedback: List[Dict[str, str]]
    improvement_areas: List[str]

def generate_unique_questions() -> List[str]:
    """Generate unique dermatology questions based on the content."""
    prompt = """Generate 5 unique and challenging dermatology questions from test deep understanding of clinical concepts. 
    Questions should:
    - Focus on different aspects of dermatology (diagnosis, treatment, pathology, etc.)
    - generate questions from docs-content-only
    - Test application of knowledge rather than mere recall
    - Cover diverse skin conditions and treatments
    - Be clinically relevant and practice-oriented
    
    Format: Return only a JSON array of 5 question strings.
    Example question style: "Explain the pathophysiology of psoriasis and how this relates to the mechanism of action of biological treatments."
    """
    
    try:
        # Get relevant document context
        relevant_docs = quiz_storage.vector_store.similarity_search(prompt, k=5)
        doc_context = "\n\n".join([d.page_content for d in relevant_docs])

        # Generate questions using both prompt and document context
        full_prompt = f"Document Context:\n{doc_context}\n\nTask: {prompt}"
        response = quiz_storage.qa_chain.run(full_prompt)

        # Clean and parse response
        response = response.replace("'", '"').strip()
        if '[' in response and ']' in response:
            response = response[response.find('['):response.rfind(']')+1]
        questions = json.loads(response)

        # Validate questions against documents
        validated_questions = []
        for question in questions[:5]:
            # Verify question is answerable from documents
            supporting_docs = quiz_storage.vector_store.similarity_search(question, k=1)
            if supporting_docs:
                validated_questions.append(question)
        
        return validated_questions[:5]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {str(e)}"
        )

logging.basicConfig(
    filename='app_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize PDFs at startup

@app.on_event("startup")
async def startup_event():
    pdf_dir = "datamn"
    try:
        documents = []
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
                documents.extend(loader.load())
        
        # Split documents with smaller chunks for better question generation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store with stronger embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-large"
        )
        quiz_storage.vector_store = FAISS.from_documents(splits, embeddings)
        
        # Create QA chain with document focus
        llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY
        )
        quiz_storage.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=quiz_storage.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"verbose": True}
        )
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        raise


# Get API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        audio_content = await audio.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name

        # Convert audio to WAV format if needed
        audio = AudioSegment.from_file(temp_audio_path)
        audio.export(temp_audio_path, format="wav")
        
        # Read the processed audio file and encode it to base64
        with open(temp_audio_path, "rb") as audio_file:
            content = base64.b64encode(audio_file.read()).decode("utf-8")

        # Prepare the request to Google Cloud Speech-to-Text API
        url = "https://speech.googleapis.com/v1/speech:recognize"
        
        headers = {
            "Authorization": f"Bearer {GOOGLE_API_KEY}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        data = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": audio.frame_rate,
                "languageCode": "en-US",
                "enableAutomaticPunctuation": True
            },
            "audio": {
                "content": content
            }
        }

        # Make the request to Google Cloud
        response = requests.post(
            f"{url}?key={GOOGLE_API_KEY}",
            headers=headers,
            json=data
        )

        # Clean up temporary file
        os.unlink(temp_audio_path)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Google Cloud API error: {response.text}"
            )

        result = response.json()
        
        # Extract the transcribed text
        if "results" in result and len(result["results"]) > 0:
            transcribed_text = result["results"][0]["alternatives"][0]["transcript"]
            return {"text": transcribed_text}
        else:
            return {"text": ""}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/start-quiz/{user_id}")
async def start_quiz(user_id: str):
    if not quiz_storage.qa_chain:
        raise HTTPException(status_code=500, detail="Quiz system not properly initialized")
    
    # Generate unique questions for this session
    questions = generate_unique_questions()
    
    quiz_storage.user_sessions[user_id] = {
        "current_question": 0,
        "answers": [],
        "scores": [],
        "feedback": [],
        "questions": questions
    }
    
    return get_next_question(user_id)
@app.post("/speech-to-text")
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


@app.post("/answer-question/{user_id}")
async def answer_question(user_id: str, response: QuizResponse):
    if user_id not in quiz_storage.user_sessions:
        raise HTTPException(status_code=404, detail="Quiz session not found")
    
    session = quiz_storage.user_sessions[user_id]
    
    if session["current_question"] >= 5:
        raise HTTPException(status_code=400, detail="Quiz already completed")
    
    current_question = session["questions"][session["current_question"]]
    
    # Handle skipped questions
    if response.answer == "SKIPPED":
        ideal_answer_prompt = f"""
        Based on the content, provide a detailed ideal answer for this question:
        Question: {current_question}
        
        Format your response as a clear, concise explanation that could serve as a model answer.
        """
        
        ideal_answer = quiz_storage.qa_chain.run(ideal_answer_prompt)
        
        session["answers"].append("SKIPPED")
        session["scores"].append(0)
        session["feedback"].append({
            "question": current_question,
            "feedback": "Question was skipped",
            "improvement": "Study the provided ideal answer",
            "ideal_answer": ideal_answer
        })
        session["current_question"] += 1
        
        if session["current_question"] >= 5:
            return calculate_results(user_id)
        else:
            return get_next_question(user_id)
    
    # Evaluate non-skipped answers
    evaluation_prompt = f"""
    Evaluate this answer to the following question:
    Question: {current_question}
    Answer: {response.answer}
    
    Provide:
    1. A score from 0 to 1 based on accuracy and completeness
    2. Specific feedback on the answer
    3. One key area for improvement
    
    Format: Return a JSON object with keys 'score', 'feedback', and 'improvement'
    """
    
    try:
        eval_response = quiz_storage.qa_chain.run(evaluation_prompt)
        eval_data = json.loads(eval_response)
        
        session["answers"].append(response.answer)
        session["scores"].append(float(eval_data["score"]))
        session["feedback"].append({
            "question": current_question,
            "feedback": eval_data["feedback"],
            "improvement": eval_data["improvement"]
        })
        session["current_question"] += 1
        
        if session["current_question"] >= 5:
            return calculate_results(user_id)
        else:
            return get_next_question(user_id)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating answer: {str(e)}"
        )

@app.post("/process-voice")
async def process_voice(voice_input: VoiceInput):
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(voice_input.audio_data)
        audio_file = io.BytesIO(audio_bytes)
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            
            # Attempt to recognize speech
            text = recognizer.recognize_google(audio)
            print(text)
            
            return {"success": True, "text": text}
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing voice input: {str(e)}"
        )

def get_next_question(user_id: str) -> Dict:
    session = quiz_storage.user_sessions[user_id]
    return {
        "question_number": session["current_question"] + 1,
        "question_text": session["questions"][session["current_question"]]
    }

def calculate_results(user_id: str) -> QuizResult:
    session = quiz_storage.user_sessions[user_id]
    
    total_score = sum(session["scores"])
    
    return QuizResult(
        total_score=total_score,
        max_score=5.0,
        feedback=session["feedback"],
        improvement_areas=[fb["improvement"] for fb in session["feedback"]]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)