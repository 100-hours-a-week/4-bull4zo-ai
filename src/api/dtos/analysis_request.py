from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    start_date: str
    end_date: str
