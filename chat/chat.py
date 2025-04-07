from fastapi import APIRouter, HTTPException
from typing import List
from database import SessionLocal
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from .models import ChatHistory, ChatEmbedding
from .chat_request import ChatRequest
from sqlalchemy import text
import openai
import os

router = APIRouter()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Researcher agent
researcher = Agent(
        name="Researcher",
        tools=[DuckDuckGoTools()],
        description="A specialized agent that performs web research on topics. It uses DuckDuckGo to search the web and gather information.",
        model=OpenAIChat(id='gpt-4o'),
        show_tool_calls=True
    )

# Create agent instances
mcp_agent = Team(
    name="Support Team",
    mode="coordinate",
    model=OpenAIChat("gpt-4o"),
    members=[researcher],
    instructions=[
        "Utilize a specialized agent to find information or perform tasks when needed.",
        "Utilize the researcher agent to perform web research when needed.",
        "If the user asks for information about the company, use the researcher agent to find information.",
        "If the user asks for information about the product, use the researcher agent to find information.",
        "If the user asks for information about the team, use the researcher agent to find information.",
        "If the user asks for information about the roadmap, use the researcher agent to find information.",
        "If the user asks for information about the pricing, use the researcher agent to find information.",
        "If the user asks for information about the FAQ, use the researcher agent to find information.",
        "If the user asks for information about the terms of service, use the researcher agent to find information.",
        "If the user asks for information about the privacy policy, use the researcher agent to find information.",
        "If the user asks for information about the refund policy, use the researcher agent to find information.",
        "If the user asks for information about the contact information, use the researcher agent to find information.",
        "If the user asks for information about the support, use the researcher agent to find information.",
        "If the user asks for information about the help, use the researcher agent to find information.",
        "If the user asks for information about the billing, use the researcher agent to find information.",
        "If the user asks for information about the account, use the researcher agent to find information.",
        "If the user asks for information about the subscription, use the researcher agent to find information.",
        "If the user asks for information about the payment, use the researcher agent to find information.",
        "If the user asks for information about the invoice, use the researcher agent to find information.",
        "If the user asks for information about the receipt, use the researcher agent to find information.",
        "If the user asks for information about the order, use the researcher agent to find information.",
        "If the user asks for information about the shipping, use the researcher agent to find information.",   
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)

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

# Function to handle research requests
async def handle_research(query: str) -> str:
    return researcher.run(query).content

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
        
        # Check if the message requires research
        if "research" in request.message.lower() or "find information about" in request.message.lower():
            research_result = await handle_research(request.message)
            full_message = f"{context}Research results: {research_result}\n\nCurrent message: {request.message}"
        
        # Get response from agent
        response = mcp_agent.run(full_message).content
        
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