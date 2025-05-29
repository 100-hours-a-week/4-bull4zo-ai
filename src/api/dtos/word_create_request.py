from pydantic import BaseModel
from typing import Optional

class WordCreateRequest(BaseModel):
    word: str
