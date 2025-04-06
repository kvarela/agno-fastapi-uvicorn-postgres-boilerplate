from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector
from database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_input = Column(Text)
    agent_response = Column(Text)

class ChatEmbedding(Base):
    __tablename__ = "chat_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    embedding = Column(Vector(dim=1536))  # OpenAI embeddings are 1536-dimensional
