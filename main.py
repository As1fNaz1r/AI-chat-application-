from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from pdf_processor import process_pdfs
from rag_system import RAGSystem

app = FastAPI()

rag_system = None

class Query(BaseModel):
    question: str

@app.post("/upload-pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global rag_system
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        pdf_contents = [await file.read() for file in files]
        knowledge_base = process_pdfs(pdf_contents)
        rag_system = RAGSystem(knowledge_base)
        return {"message": "PDFs processed successfully", "file_count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@app.post("/query/")
async def query(query: Query):
    if not rag_system:
        raise HTTPException(status_code=400, detail="Please upload PDFs first")
    
    try:
        response = rag_system.get_answer(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chat API. Use /upload-pdfs/ to upload PDFs and /query/ to ask questions."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)