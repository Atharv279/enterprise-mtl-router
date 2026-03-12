from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID, uuid4

class Complaint(BaseModel):
    complaint_id: UUID = Field(default_factory=uuid4) # Primary deterministic identifier
    timestamp: datetime = Field(default_factory=datetime.utcnow) # Chronological marker
    raw_text: str # Unmodified multilingual narrative
    media_path: Optional[str] = None # Local absolute directory path
    transcribed_text: Optional[str] = None # Output from localized ASR
    semantic_vector: Optional[List[float]] = None # 1024-dimensional E5 embedding
    predicted_priority: Optional[int] = None # Argmax output (1=High, 2=Medium, 3=Low)
    estimated_eta: Optional[float] = None # Continuous output from regression head
    assigned_officer: Optional[UUID] = None # Foreign key post-optimization

class Officer(BaseModel):
    officer_id: UUID = Field(default_factory=uuid4) # Primary deterministic identifier
    expertise_profile: str # Textual encapsulation of skills
    skill_vector: List[float] # Pre-computed 1024-dimensional embedding
    current_load: int = 0 # Dynamic tracker for assignment capacity