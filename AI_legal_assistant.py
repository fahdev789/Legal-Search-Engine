from fastapi import FastAPI, Request, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from typing import Optional
import shutil

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api  # Set the environment variable

app = FastAPI()

# Constants and configuration
persist_directory = "faissembeddings"
model_docs = "sentence-transformers/all-mpnet-base-v2"

# Load embeddings
hf_model = HuggingFaceEmbeddings(model_name=model_docs)
embeddings_instance = FAISS.load_local(
    folder_path=persist_directory,
    embeddings=hf_model,
    allow_dangerous_deserialization=True
)

# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print('Request for index page received')
    return templates.TemplateResponse('index.html', {"request": request})

@app.get('/favicon.ico')
async def favicon():
    file_name = 'favicon.ico'
    file_path = './static/' + file_name
    return FileResponse(path=file_path, headers={'mimetype': 'image/vnd.microsoft.icon'})

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB file size limit
@app.post("/upload")
async def add_document(file: UploadFile):
    try:
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File is too large.")
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp", file.filename)
        os.makedirs("temp", exist_ok=True)

        # Save file
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        # Log file path
        logging.info(f"File saved at: {temp_file_path}")
        
        # Handle different file types
        try:
            if file.content_type == "text/plain":
                loader = TextLoader(temp_file_path)
            elif file.content_type == "application/pdf":
                loader = PyPDFLoader(temp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type.")
        except Exception as e:
            logging.error(f"Error loading file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")

        docs = loader.load()

        # Split the documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.split_documents(docs)
        
        # Add documents to embeddings
        embeddings_instance.add_documents(documents)
        embeddings_instance.save_local(folder_path=persist_directory)

        # Clean up the temporary file
        os.remove(temp_file_path)

        logging.info(f"Document {file.filename} added to embeddings successfully.")
        return JSONResponse(content={"message": "Document successfully added to embeddings."})
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/chat")
async def ask_question(
    user_prompt: str = Form(...),
    selected_model: str = Form("llama-3.1-8b-instant"),
    temperature: float = Form(0.5),
    p_value: float = Form(0.9)
):
    try:
        # Initialize ChatGroq model
        chat_model = ChatGroq(
            api_key=groq_api,
            model=selected_model,
            temperature=temperature,
            top_p=p_value
        )

        # Retrieve relevant documents from embeddings
        retriever = embeddings_instance.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(user_prompt)

        # Concatenate retrieved documents for context
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Generate response
        messages = [
            SystemMessage(content="You are a helpful assistant specialized in contracts."),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {user_prompt}")
        ]
        response = chat_model(messages)
        return JSONResponse(content={"response": response.content})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("AI_legal_assistant:app", port=8000, log_level="debug")
