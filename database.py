import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/chatdb")
db_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
Base = declarative_base()

# Enable pgvector extension and create indexes
def init_db():
    with db_engine.connect() as conn:
        try:
            # Enable the vector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            
            # Verify vector extension is installed
            extension_exists = conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            ).scalar()
            
            if not extension_exists:
                raise Exception("pgvector extension could not be installed")
            
            # Create all tables if they don't exist
            Base.metadata.create_all(bind=db_engine)
            conn.commit()
            
            # Verify table exists before creating index
            table_exists = conn.execute(
                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'chat_embeddings')")
            ).scalar()
            
            if table_exists:
                # Create the HNSW index if it doesn't exist
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS chat_embeddings_embedding_idx 
                    ON chat_embeddings 
                    USING hnsw (embedding vector_cosine_ops)
                """))
                conn.commit()
            
            print("Database initialization completed successfully")
            
        except Exception as e:
            print(f"Error during database initialization: {e}")
            conn.rollback()
            raise

# Initialize database
init_db()