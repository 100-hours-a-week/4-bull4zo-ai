from pydantic import BaseModel
from typing import Literal, Optional
 
class VotePublishResponse(BaseModel):
    status: Literal["published", "blocked"]
    vote_id: Optional[int] = None
    reason: Optional[str] = None 