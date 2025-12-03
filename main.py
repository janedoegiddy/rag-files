import os
import shutil
import boto3
from datetime import datetime
from typing import List, Tuple
import psycopg2
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
import google.generativeai as genai

# --- CONFIGURATION & CLIENTS ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
SSM_PARAM_NAME_GEMINI = os.environ.get("SSM_PARAM_NAME_GEMINI")

# --- Database Configuration ---
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

DOCUMENTS_S3_PREFIX = "documents/"
LOGS_S3_PREFIX = "logs/"

s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
ssm_client = boto3.client("ssm", region_name=os.environ.get("AWS_REGION"))

# --- GEMINI CLIENT SETUP ---
def get_gemini_api_key():
    """Fetches the Gemini API Key from SSM or falls back to an environment variable."""
    try:
        # First, try to get the key from AWS SSM (for production on EC2)
        print(f"Attempting to fetch Gemini API key from SSM parameter: {SSM_PARAM_NAME_GEMINI}")
        response = ssm_client.get_parameter(Name=SSM_PARAM_NAME_GEMINI, WithDecryption=True)
        print("Successfully fetched API key from SSM.")
        return response['Parameter']['Value']
    except Exception as e:
        print(f"Could not retrieve API key from SSM: {e}")
        # If SSM fails, fall back to environment variable (for local Docker testing)
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print("Falling back to GEMINI_API_KEY environment variable.")
            return api_key
        else:
            print("ERROR: Gemini API key not found in SSM or environment variables.")
            raise e

try:
    gemini_api_key = get_gemini_api_key()
    genai.configure(api_key=gemini_api_key)
    embedding_model = "text-embedding-004"  # Google's embedding model
    llm_model_name = "gemini-1.5-pro-latest"      # Google's powerful model
    print("Google Gemini client configured successfully.")
except Exception as e:
    llm_model_name = None
    print(f"FATAL: Failed to configure Gemini client: {e}")


# --- DATABASE CONNECTION ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    register_vector(conn)
    return conn

# --- HELPER FUNCTIONS ---
def simple_text_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size: return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def extract_text_from_pdf_path(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def build_prompt(question: str, context_chunks: List[str], chat_history: List[Tuple[str, str]] = []) -> str:
    history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
    context_str = "\n---\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following pieces of context and the chat history to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Chat History:
{history_str}

Context:
{context_str}

Question: {question}
Answer:"""

def log_conversation(user_id: str, question: str, answer: str):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_content = f"Timestamp: {timestamp}\nUser: {question}\nAgent: {answer}\n---"
    log_key = f"{LOGS_S3_PREFIX}{user_id}/{timestamp}.log"
    try:
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=log_key, Body=log_content.encode('utf-8'))
    except Exception as e:
        print(f"Error logging conversation to S3: {e}")


# --- FASTAPI APPLICATION ---
TEMP_DIR = "/tmp/uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

app = FastAPI(title="RAG API (Gemini 1.5 Pro + pgvector)")

class QuestionRequest(BaseModel):
    user_id: str
    question: str
    chat_history: List[Tuple[str, str]] = []

@app.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    local_file_path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(local_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        s3_doc_key = f"{DOCUMENTS_S3_PREFIX}{user_id}/{file.filename}"
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_doc_key)

        text = extract_text_from_pdf_path(local_file_path)
        if not text:
            return {"filename": file.filename, "status": "No text could be extracted from the PDF."}

        chunks = simple_text_splitter(text)
        if not chunks:
            return {"filename": file.filename, "status": "No text chunks generated."}

        print(f"Generating embeddings for {len(chunks)} chunks with Gemini...")
        response = genai.embed_content(model=embedding_model, content=chunks, task_type="retrieval_document")
        embeddings = response['embedding']

        conn = get_db_connection()
        cur = conn.cursor()
        for chunk, embedding in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO documents (user_id, filename, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                (user_id, file.filename, chunk, embedding)
            )
        conn.commit()
        cur.close()
        conn.close()

        return {"filename": file.filename, "status": f"Processed and added {len(chunks)} chunks to the database."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        print(f"Embedding question for user: {request.user_id}")
        response = genai.embed_content(model=embedding_model, content=request.question, task_type="retrieval_query")
        question_embedding = response['embedding']

        conn = get_db_connection()
        cur = conn.cursor()
        # The <=> operator calculates the cosine distance for similarity search
        cur.execute(
            "SELECT chunk_text FROM documents WHERE user_id = %s ORDER BY embedding <=> %s LIMIT 5",
            (request.user_id, str(question_embedding))
        )
        relevant_chunks = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        if not relevant_chunks:
            return {"answer": "I could not find any relevant information in your documents to answer that question.", "source_documents": []}

        print("Generating answer with Gemini...")
        prompt = build_prompt(request.question, relevant_chunks, request.chat_history)
        model = genai.GenerativeModel(llm_model_name)
        llm_response = model.generate_content(prompt)
        answer = llm_response.text

        log_conversation(request.user_id, request.question, answer)
        return {"answer": answer, "source_documents": relevant_chunks}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
