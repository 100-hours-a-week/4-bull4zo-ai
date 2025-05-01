from pydantic import BaseModel

class ModerationResultRequest(BaseModel):
    voteId: int
    result: str
    reason: str
    reasonDetail: str
    version: str
