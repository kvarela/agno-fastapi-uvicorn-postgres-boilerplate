from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/chatdb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define database models
class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_input = Column(Text)
    agent_response = Column(Text)

# Create tables
Base.metadata.create_all(bind=engine)

# Define request model
class ChatRequest(BaseModel):
    message: str
    include_history: Optional[bool] = True

# Initialize Agno agent
def create_agent():
    return Agent(
        name="chat_agent",
        tools=[DuckDuckGoTools()],
        description="A chat agent that can answer questions and help with tasks",
        model=OpenAIChat(id='gpt-4o'),
        show_tool_calls=True
    )

# Create agent instance
agent = create_agent()

def get_recent_chat_history(db, limit: int = 5) -> List[dict]:
    """Get recent chat history from the database."""
    history = db.query(ChatHistory).order_by(ChatHistory.id.desc()).limit(limit).all()
    return [
        {
            "user": entry.user_input,
            "assistant": entry.agent_response
        }
        for entry in reversed(history)
    ]

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        db = SessionLocal()
        
        # Get context from previous conversations if requested
        context = ""
        if request.include_history:
            history = get_recent_chat_history(db)
            if history:
                context = "Previous conversation:\n"
                for entry in history:
                    context += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
        
        # Combine context with current message
        full_message = f"{context}Current message: {request.message}"
        
        # Get response from agent
        response = agent.run(full_message).content
        print(f"Response: {response}")
        
        # Store in database
        chat_entry = ChatHistory(
            user_input=request.message,
            agent_response=response
        )
        db.add(chat_entry)
        db.commit()
        db.close()
        
        return {"response": response}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 