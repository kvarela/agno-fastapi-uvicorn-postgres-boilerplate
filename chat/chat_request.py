from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    include_history: Optional[bool] = True 