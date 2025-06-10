from pydantic import BaseModel

class ModelVoteRequest(BaseModel):
    wordId: int
    content: str
    imageUrl: str = ""
    imageName: str = ""
    openAt: str
    closedAt: str
    version: str
