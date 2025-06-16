from pydantic import BaseModel

class ModelVoteRequest(BaseModel):
    content: str
    imageUrl: str = ""
    imageName: str = ""
    openAt: str
    closedAt: str
    version: str
