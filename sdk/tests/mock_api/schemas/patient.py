from typing import Optional
from pydantic import BaseModel


class Patient(BaseModel):
    id: int
    mrn: int
    gender: Optional[str] = None
    dob: int
    first_name: Optional[str] = None  # needs to be optional because there's one patient in the database with no first name
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    first_seen: Optional[int] = None
    last_updated: Optional[int] = None
    source_id: Optional[int] = None
    height: Optional[float] = None
    height_units: Optional[str] = None
    height_time: Optional[int] = None
    weight: Optional[float] = None
    weight_units: Optional[str] = None
    weight_time: Optional[int] = None
