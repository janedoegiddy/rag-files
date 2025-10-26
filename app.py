import os
import shutil
import boto3
import numpy as np
from datetime import datetime
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from mangum import Mangum
from pydantic import BaseModel

from openai import OpenAI
from pypdf import PdfReader

# --- CONFIGURATION & CLIENTS ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
SSM_PARAM_NAME = os.environ.get("SSM_PARAM_NAME")
DOCUMENTS_S3_PREFIX = "documents/"
LOGS_S3_PREFIX = "logs/"

s3_client = boto3.client("s3")
ssm_client = boto3.client("ssm")


def get_openai_api_key(): # Renamed for clarity
    """Fetches the OpenAI API Key from SSM Parameter Store."""
    try:
        response = ssm_client.get_parameter(Name=SSM_PARAM_NAME, WithDecryption=True)
        return response['Parameter']['Value']
    except Exception as e:
        print(f"ERROR: Could not retrieve API key from SSM: {e}")
        raise e

# --- CHANGE: Initialize OpenAI client on cold start ---
try:
    openai_client = OpenAI(api_key=get_openai_api_key())
    embedding_model = "text-embedding-3-small"
    llm_model = "gpt-4o-mini"
except Exception as e:
    print(f"FATAL: Could not initialize OpenAI client. Error: {e}")
    openai_client = None

# --- UNCHANGED HELPER FUNCTIONS ---
def simple_text_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size: return [text]
    chunks = []
    start = 0
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
            text += page.extract_text() or ""
    return text

def find_similar_documents(query_embedding: np.ndarray, doc_embeddings: np.ndarray, documents: List[str], top_k: int = 5) -> List[str]:
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    similarities = np.dot(doc_norms, query_norm)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def build_prompt(question: str, context_chunks: List[str], chat_history: List[Tuple[str, str]] = []) -> str:
    history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
    context_str = "\n---\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following pieces of context and the chat history to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Chat History:
{history_str}
Context:
{context_str}
Question: {question}
Answer:"""
    return prompt

def log_conversation(user_id: str, question: str, answer: str):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_content = f"Timestamp: {timestamp}\nUser: {question}\nAgent: {answer}\n---"
    log_key = f"{LOGS_S3_PREFIX}{user_id}/{timestamp}.log"
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=log_key, Body=log_content.encode('utf-8'))

# --- FastAPI Application ---
TEMP_DIR = "/tmp/uploads"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
app = FastAPI(title="Lightweight RAG Lambda API (OpenAI)")
class QuestionRequest(BaseModel):
    user_id: str
    question: str
    chat_history: List[Tuple[str, str]] = []

@app.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    if not user_id: raise HTTPException(status_code=400, detail="user_id is required.")
    s3_key = f"{DOCUMENTS_S3_PREFIX}{user_id}/{file.filename}"
    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET_NAME, s3_key)
        return {"filename": file.filename, "user_id": user_id, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Steps 1 & 2: List, download, extract text, and split into chunks (UNCHANGED)
        user_docs_prefix = f"{DOCUMENTS_S3_PREFIX}{request.user_id}/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=user_docs_prefix)
        if 'Contents' not in response: raise HTTPException(status_code=404, detail="No documents found for this user.")
        all_chunks = []
        for obj in response['Contents']:
            file_key = obj['Key']
            local_file_path = os.path.join(TEMP_DIR, os.path.basename(file_key))
            s3_client.download_file(S3_BUCKET_NAME, file_key, local_file_path)
            text = extract_text_from_pdf_path(local_file_path)
            chunks = simple_text_splitter(text)
            all_chunks.extend(chunks)
            os.remove(local_file_path)
        if not all_chunks: return {"answer": "No content found in documents.", "source_documents": []}

        # --- CHANGE: Step 3: Generate embeddings using OpenAI ---
        print(f"Generating embeddings for {len(all_chunks)} chunks with OpenAI...")
        doc_embeddings_response = openai_client.embeddings.create(model=embedding_model, input=all_chunks)
        question_embedding_response = openai_client.embeddings.create(model=embedding_model, input=request.question)
        
        # Extract the vector from the response objects
        doc_embeddings = np.array([item.embedding for item in doc_embeddings_response.data])
        question_embedding = np.array(question_embedding_response.data[0].embedding)

        # Step 4: Find relevant chunks using NumPy (UNCHANGED)
        relevant_chunks = find_similar_documents(question_embedding, doc_embeddings, all_chunks)
        
        # --- CHANGE: Step 5: Build prompt and get answer from OpenAI LLM ---
        prompt = build_prompt(request.question, relevant_chunks, request.chat_history)
        llm_response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        answer = llm_response.choices[0].message.content

        # Step 6: Log and respond (UNCHANGED)
        log_conversation(request.user_id, request.question, answer)
        return {"answer": answer, "source_documents": relevant_chunks}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mangum handler (UNCHANGED)
handler = Mangum(app)
