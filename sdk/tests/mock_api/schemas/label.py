from pydantic import BaseModel
from typing import Optional, List


class LabelName(BaseModel):
    id: int
    name: str
    parent_id: Optional[int] = None
    parent_name: Optional[str] = None


class LabelSource(BaseModel):
    id: int
    name: str
    description: Optional[str] = None


class Label(BaseModel):
    label_entry_id: int
    label_name_id: int
    label_name: str
    requested_name_id: Optional[int] = None
    requested_name: Optional[str] = None
    device_id: int
    device_tag: str
    patient_id: Optional[int] = None
    mrn: Optional[int] = None
    start_time_n: int
    end_time_n: int
    label_source_id: Optional[int] = None
    label_source: Optional[str] = None


class LabelsQuery(BaseModel):
    label_name_id_list: Optional[List[int]] = None
    name_list: Optional[List[str]] = None
    device_list: Optional[List[int | str]] = None
    patient_id_list: Optional[List[int]] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    time_units: Optional[str] = None
    include_descendants: Optional[bool] = True
    limit: Optional[int] = 1000
    offset: Optional[int] = 0
