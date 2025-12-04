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
from openai import OpenAI # --- OPENAI ---

# --- CONFIGURATION & CLIENTS ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
# --- OPENAI ---: Use an SSM parameter name for the OpenAI key
SSM_PARAM_NAME_OPENAI = os.environ.get("SSM_PARAM_NAME_OPENAI")

# --- Database Configuration ---
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

DOCUMENTS_S3_PREFIX = "documents/"
LOGS_S3_PREFIX = "logs/"

s3_client = boto3.client("s3")
ssm_client = boto3.client("ssm")

# --- OPENAI ---: Function to fetch the API key from SSM
def get_openai_api_key():
    try:
        print(f"Attempting to fetch OpenAI API key from SSM parameter: {SSM_PARAM_NAME_OPENAI}")
        response = ssm_client.get_parameter(Name=SSM_PARAM_NAME_OPENAI, WithDecryption=True)
        return response['Parameter']['Value']
    except Exception as e:
        print(f"Could not retrieve API key from SSM: {e}. Falling back to OPENAI_API_KEY env var.")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: raise e
        return api_key

# --- OPENAI ---: Configure the OpenAI client
try:
    api_key = get_openai_api_key()
    openai_client = OpenAI(api_key=api_key)
    embedding_model = "text-embedding-3-small"
    llm_model = "gpt-4o-mini"
    print("OpenAI client configured successfully.")
except Exception as e:
    openai_client = None
    print(f"FATAL: Failed to configure OpenAI client: {e}")

# (The rest of the file, from get_db_connection() onwards, is identical to the Gemini version,
#  but the parts using the AI models inside the endpoints need to be changed.)

# --- DATABASE CONNECTION ---
def get_db_connection():
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    register_vector(conn)
    return conn

# --- HELPER FUNCTIONS (UNCHANGED) ---
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
            if extracted_text: text += extracted_text
    return text

def build_prompt(question: str, context_chunks: List[str], chat_history: List[Tuple[str, str]] = []) -> str:
    history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
    context_str = "\n---\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following pieces of context...
Question: {question}
Answer:""" # Abridged for brevity

def log_conversation(user_id: str, question: str, answer: str):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_content = f"Timestamp: {timestamp}\nUser: {question}\nAgent: {answer}\n---"
    log_key = f"{LOGS_S3_PREFIX}{user_id}/{timestamp}.log"
    try:
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=log_key, Body=log_content.encode('utf-8'))
    except Exception as e:
        print(f"Error logging conversation to S3: {e}")

# --- FastAPI Application ---
TEMP_DIR = "/tmp/uploads"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
app = FastAPI(title="RAG API (OpenAI + pgvector)")

class QuestionRequest(BaseModel):
    user_id: str
    question: str
    chat_history: List[Tuple[str, str]] = []

@app.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    local_file_path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(local_file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        s3_doc_key = f"{DOCUMENTS_S3_PREFIX}{user_id}/{file.filename}"
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_doc_key)
        text = extract_text_from_pdf_path(local_file_path)
        chunks = simple_text_splitter(text)
        if not chunks: return {"filename": file.filename, "status": "No text chunks generated."}
        
        # --- OPENAI ---: Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks with OpenAI...")
        response = openai_client.embeddings.create(model=embedding_model, input=chunks)
        embeddings = [item.embedding for item in response.data]

        conn = get_db_connection()
        cur = conn.cursor()
        for chunk, embedding in zip(chunks, embeddings):
            cur.execute("INSERT INTO documents (user_id, filename, chunk_text, embedding) VALUES (%s, %s, %s, %s)", (user_id, file.filename, chunk, embedding))
        conn.commit()
        cur.close()
        conn.close()
        return {"filename": file.filename, "status": f"Processed and added {len(chunks)} chunks."}
    finally:
        if os.path.exists(local_file_path): os.remove(local_file_path)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # --- OPENAI ---: Embed the question
    response = openai_client.embeddings.create(model=embedding_model, input=request.question)
    question_embedding = response.data[0].embedding
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT chunk_text FROM documents WHERE user_id = %s ORDER BY embedding <=> %s LIMIT 5", (request.user_id, str(question_embedding)))
    relevant_chunks = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    if not relevant_chunks: return {"answer": "No relevant information found.", "source_documents": []}
    
    # --- OPENAI ---: Get answer from the LLM
    prompt = build_prompt(request.question, relevant_chunks, request.chat_history)
    llm_response = openai_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    answer = llm_response.choices[0].message.content
    
    log_conversation(request.user_id, request.question, answer)
    return {"answer": answer, "source_documents": relevant_chunks}

@app.get("/documents/{user_id}")
async def get_user_documents(user_id: str):
    """Retrieves a list of unique document filenames for a given user_id."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Query for distinct filenames for the given user, ordered by name
        cur.execute(
            "SELECT DISTINCT filename FROM documents WHERE user_id = %s ORDER BY filename",
            (user_id,)
        )
        
        # The result from the DB is a list of tuples, e.g., [('doc1.pdf',), ('doc2.pdf',)]
        # We flatten it into a simple list of strings.
        documents = [row[0] for row in cur.fetchall()]
        cur.close()
        
        return {"documents": documents}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching documents: {e}")
    finally:
        if conn:
            conn.close()
