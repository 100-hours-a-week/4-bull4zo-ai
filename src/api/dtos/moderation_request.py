from pydantic import BaseModel

class ModerationRequest(BaseModel):
    voteContent: str
