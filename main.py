from fastapi import FastAPI
from dotenv import load_dotenv
from health import router as health_router
from chat.chat import router as chat_router
from content import router as content_router
from database import init_db  # Import database initialization

# Load environment variables
load_dotenv()

# Initialize database and create tables
init_db()

# Initialize FastAPI app
app = FastAPI()
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(content_router)