from fastapi import APIRouter, HTTPException
from typing import List
from database import SessionLocal
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from .models import ChatHistory, ChatEmbedding
from .chat_request import ChatRequest
from sqlalchemy import text
import json
import openai
import os

router = APIRouter()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def get_embedding(text: str) -> list:
    """Get embedding for text using OpenAI's API."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def get_relevant_memories(db, query: str, limit: int = 3) -> List[dict]:
    """Get relevant memories using vector similarity search."""
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Perform vector similarity search using HNSW index
    memories = db.execute(
        text("""
            SELECT text, 
                   (embedding <-> CAST(:embedding AS vector)) as distance
            FROM chat_embeddings
            ORDER BY embedding <-> CAST(:embedding AS vector)
            LIMIT :limit
        """),
        {"embedding": query_embedding, "limit": limit}
    ).fetchall()
    
    return [
        {
            "text": memory.text,
            "similarity": 1 - float(memory.distance)  # Convert distance to similarity
        }
        for memory in memories
    ]

def store_memory(db, text: str):
    """Store a new memory with its embedding."""
    # Get embedding for the text
    embedding = get_embedding(text)
    
    # Create new memory entry
    memory = ChatEmbedding(
        text=text,
        embedding=embedding,  # OpenAI embeddings are already in list format
    )
    
    db.add(memory)
    db.commit()

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        db = SessionLocal()
        
        # Get relevant memories if requested
        context = ""
        if request.include_history:
            memories = get_relevant_memories(db, request.message)
            if memories:
                context = "Relevant memories:\n"
                for memory in memories:
                    context += f"- {memory['text']} (similarity: {memory['similarity']:.2f})\n"
                context += "\n"
        
        # Combine context with current message
        full_message = f"{context}Current message: {request.message}"
        
        # Get response from agent
        response = agent.run(full_message).content
        
        # Store the conversation in both chat history and memory
        chat_entry = ChatHistory(
            user_input=request.message,
            agent_response=response
        )
        db.add(chat_entry)
        db.flush()  # Get the ID
        
        # Store as memory
        store_memory(
            db,
            text=f"User: {request.message}\nAssistant: {response}",
        )
        
        db.commit()
        db.close()
        
        return {"response": response}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))