from pydantic import BaseModel

class ModerationRequest(BaseModel):
    voteId: int
    content: str
