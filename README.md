# Chat Application with FastAPI and Agno

This is a FastAPI application boilerplate that provides a chat endpoint powered by Agno agents and stores conversation history in PostgreSQL with pgvector.

I've added uvicorn for concurrency. It uses OpenAI ChatGPT for the model, but you can swap it out as you please.

## Prerequisites

- Docker and Docker Compose
- OpenAI API key

## Setup

1. Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key_here
```

2. Build and start the services:

```bash
docker-compose up --build
```

In the app container, run `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

The application will be available at `http://localhost:8000`

## API Endpoints

### POST /chat

Send a chat message to the agent.

Request body:

```json
{
  "message": "Your message here"
}
```

### GET /health

Check the health status of the application.

## Features

- FastAPI async endpoint for chat
- Agno agent with sub-agents for research and support
- PostgreSQL with pgvector for data storage
- Dockerized environment
