from fastapi import APIRouter, UploadFile, File, HTTPException
import PyPDF2
import io
from database import SessionLocal
from .models import ChatEmbedding
from .chat import get_embedding

router = APIRouter()

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text content from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

@router.post("/content")
async def process_pdf_content(file: UploadFile = File(...)):
    try:
        # Check if the file is a PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read the PDF file
        contents = await file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf(contents)

        print(text)
        
        if not text:
            raise HTTPException(status_code=400, detail="No text content found in PDF")
        
        # Get embedding for the text
        embedding = get_embedding(text)
        
        # Store in database
        db = SessionLocal()
        try:
            content_embedding = ChatEmbedding(
                text=text,
                embedding=embedding
            )
            db.add(content_embedding)
            db.commit()
            return {"message": "PDF content processed and stored successfully"}
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}") 