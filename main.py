from fastapi import FastAPI
from typing import List
from dotenv import load_dotenv
from health import router as health_router
from chat import router as chat_router

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.include_router(health_router)
app.include_router(chat_router)