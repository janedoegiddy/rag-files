import os
import shutil
import boto3
from datetime import datetime
from typing import List, Tuple
import zipfile
import sys
import asyncio
from types import SimpleNamespace

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from mangum import Mangum
from pydantic import BaseModel
from pypdf import PdfReader
import chromadb
from openai import OpenAI # --- OPENAI ---

# --- CONFIGURATION & CLIENTS ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
# --- OPENAI ---: Use the generic SSM parameter name as requested
SSM_PARAM_NAME = os.environ.get("SSM_PARAM_NAME")

DOCUMENTS_S3_PREFIX = "documents/"
LOGS_S3_PREFIX = "logs/"
CHROMA_S3_PREFIX = "chromadb/"

s3_client = boto3.client("s3")
ssm_client = boto3.client("ssm")

# --- OPENAI ---: Function to fetch the API key from SSM
def get_openai_api_key_from_ssm():
    """Fetches the OpenAI API Key from SSM Parameter Store."""
    try:
        print(f"Fetching OpenAI API key from SSM parameter: {SSM_PARAM_NAME}")
        response = ssm_client.get_parameter(Name=SSM_PARAM_NAME, WithDecryption=True)
        return response['Parameter']['Value']
    except Exception as e:
        print(f"ERROR: Could not retrieve API key from SSM: {e}")
        raise e

# --- OPENAI ---: Configure the OpenAI client on cold start
try:
    api_key = get_openai_api_key_from_ssm()
    openai_client = OpenAI(api_key=api_key)
    # Define model names
    embedding_model = "text-embedding-3-small"
    llm_model = "gpt-4o-mini"
    print("OpenAI client configured successfully.")
except Exception as e:
    openai_client = None


# --- HELPER FUNCTIONS (UNCHANGED) ---
def simple_text_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size: return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks

def extract_text_from_pdf_path(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages: text += page.extract_text() or ""
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
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=log_key, Body=log_content.encode('utf-8'))


# --- CHROMA & S3 SYNC HELPERS (UNCHANGED) ---
CHROMA_DATA_PATH = "/tmp/chroma_db"
def sync_chroma_from_s3(user_id: str):
    s3_key, local_zip_path = f"{CHROMA_S3_PREFIX}{user_id}/chroma.zip", f"/tmp/{user_id}_chroma.zip"
    if os.path.exists(CHROMA_DATA_PATH): shutil.rmtree(CHROMA_DATA_PATH)
    os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_zip_path)
        with zipfile.ZipFile(local_zip_path, 'r') as zf: zf.extractall(CHROMA_DATA_PATH)
        os.remove(local_zip_path)
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404': raise

def sync_chroma_to_s3(user_id: str):
    local_zip_path, s3_key = f"/tmp/{user_id}_chroma.zip", f"{CHROMA_S3_PREFIX}{user_id}/chroma.zip"
    with zipfile.ZipFile(local_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(CHROMA_DATA_PATH):
            for file in files: zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), CHROMA_DATA_PATH))
    s3_client.upload_file(local_zip_path, S3_BUCKET_NAME, s3_key)
    os.remove(local_zip_path)


# --- FastAPI Application ---
TEMP_DIR = "/tmp/uploads"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
app = FastAPI(title="RAG API (OpenAI + ChromaDB)")

class QuestionRequest(BaseModel):
    user_id: str; question: str; chat_history: List[Tuple[str, str]] = []

@app.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    if not user_id: raise HTTPException(status_code=400, detail="user_id is required.")
    
    is_local_cli = isinstance(file, SimpleNamespace) and hasattr(file, 'local_path')
    if is_local_cli:
        local_file_path_for_processing = file.local_path
        filename = os.path.basename(local_file_path_for_processing)
    else:
        filename = file.filename
        local_file_path_for_processing = os.path.join(TEMP_DIR, filename)
        with open(local_file_path_for_processing, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    s3_doc_key = f"{DOCUMENTS_S3_PREFIX}{user_id}/{filename}"
    try:
        s3_client.upload_file(local_file_path_for_processing, S3_BUCKET_NAME, s3_doc_key)
    except Exception as e:
        if not is_local_cli and os.path.exists(local_file_path_for_processing):
            os.remove(local_file_path_for_processing)
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e}")

    try:
        sync_chroma_from_s3(user_id)
        client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        collection = client.get_or_create_collection(name=f"collection_{user_id.replace('-', '_')}")
        
        text = extract_text_from_pdf_path(local_file_path_for_processing)
        chunks = simple_text_splitter(text)

        if not is_local_cli:
            os.remove(local_file_path_for_processing)
        if not chunks: 
            return {"filename": filename, "status": "No text found."}

        # --- OPENAI ---: Generate embeddings for the chunks
        print(f"Generating embeddings for {len(chunks)} chunks with OpenAI...")
        response = openai_client.embeddings.create(model=embedding_model, input=chunks)
        embeddings = [item.embedding for item in response.data]
        
        doc_ids = [f"{filename}-{i}" for i in range(len(chunks))]
        collection.add(embeddings=embeddings, documents=chunks, ids=doc_ids)
        print(f"Added {len(chunks)} chunks to collection.")

        sync_chroma_to_s3(user_id)
        return {"filename": filename, "status": f"processed and added {len(chunks)} chunks."}
    except Exception as e:
        if not is_local_cli and os.path.exists(local_file_path_for_processing):
            os.remove(local_file_path_for_processing)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        sync_chroma_from_s3(request.user_id)
        client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        try:
            collection = client.get_collection(name=f"collection_{request.user_id.replace('-', '_')}")
        except ValueError: raise HTTPException(status_code=404, detail="No documents uploaded.")

        # --- OPENAI ---: Embed the question
        response = openai_client.embeddings.create(model=embedding_model, input=request.question)
        question_embedding = response.data[0].embedding

        results = collection.query(query_embeddings=[question_embedding], n_results=5)
        relevant_chunks = results['documents'][0]
        if not relevant_chunks:
            return {"answer": "No relevant information found in your documents.", "source_documents": []}
        
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
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)
