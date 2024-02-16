# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional, List, Dict, Any
from tests.mock_api.sdk_dependency import get_sdk_instance
from fastapi import APIRouter, Depends, HTTPException
from atriumdb import AtriumSDK
import tests.mock_api.schemas as schemas

router = APIRouter()


@router.post("/", response_model=List[schemas.Label])
async def search_labels(body: schemas.LabelsQuery, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):

    # make sure they arnt asking for too much data at a time
    if body.limit > 1000:
        raise HTTPException(status_code=400, detail="Limits of greater than 1000 are not allowed.")

    if body.label_name_id_list and body.name_list:
        raise HTTPException(status_code=400, detail="Only one of label_name_id_list or name_list should be provided.")

    if body.device_list and body.patient_id_list:
        raise HTTPException(status_code=400, detail="Only one of device_list or patient_id_list should be provided.")

    try:
        labels = atriumdb_sdk.get_labels(
            label_name_id_list=body.label_name_id_list,
            name_list=body.name_list,
            device_list=body.device_list,
            patient_id_list=body.patient_id_list,
            start_time=body.start_time,
            end_time=body.end_time,
            time_units=body.time_units,
            include_descendants=body.include_descendants,
            limit=body.limit, offset=body.offset
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return labels


@router.get("/source", response_model=int | schemas.LabelSource | None)
async def get_label_source(label_source_id: Optional[int] = None, label_source_name: Optional[str] = None, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):#, user: Auth0User = Security(auth.get_user)):

    if label_source_id and label_source_name:
        raise HTTPException(status_code=400, detail="Only one of label_source_id or label_source_name should be provided.")

    if label_source_id is None and label_source_name is None:
        raise HTTPException(status_code=400, detail="At least one of label_source_id or label_source_name should be provided.")

    if label_source_id is not None:
        return atriumdb_sdk.get_label_source_info(label_source_id=label_source_id)
    if label_source_name is not None:
        return atriumdb_sdk.get_label_source_id(name=label_source_name)


def validate_label_inputs(label_name_id, label_name):
    if label_name_id and label_name:
        raise HTTPException(status_code=400, detail="Only one of label_name_id or label_name should be provided.")

    if label_name_id is None and label_name is None:
        raise HTTPException(status_code=400, detail="At least one of label_name_id or label_name should be provided.")


@router.get("/name", response_model=int | schemas.LabelName | None)
async def get_label_name(label_name_id: Optional[int] = None, label_name: Optional[str] = None, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):#, user: Auth0User = Security(auth.get_user)):
    validate_label_inputs(label_name_id, label_name)

    if label_name_id is not None:
        return atriumdb_sdk.get_label_name_info(label_name_id=label_name_id)
    if label_name is not None:
        return atriumdb_sdk.get_label_name_id(name=label_name)


@router.get("/names", response_model=Dict[int, schemas.LabelName])
async def get_all_label_names(limit: int = 1000, offset: int = 0, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):#, user: Auth0User = Security(auth.get_user)):
    return atriumdb_sdk.get_all_label_names(limit=limit, offset=offset)


@router.get("/parent", response_model=schemas.LabelName | None)
async def get_label_name_parent(label_name_id: Optional[int] = None, label_name: Optional[str] = None, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    validate_label_inputs(label_name_id, label_name)
    return atriumdb_sdk.get_label_name_parent(label_name_id=label_name_id, name=label_name)


@router.get("/children", response_model=List[schemas.LabelName])
async def get_label_name_children(label_name_id: Optional[int] = None, label_name: Optional[str] = None, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    validate_label_inputs(label_name_id, label_name)
    return atriumdb_sdk.get_label_name_children(label_name_id=label_name_id, name=label_name)


@router.get("/descendents", response_model=Dict[str, Any])
async def get_all_label_name_descendents(label_name_id: Optional[int] = None, label_name: Optional[str] = None, depth: Optional[int] = None, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    validate_label_inputs(label_name_id, label_name)
    return atriumdb_sdk.get_all_label_name_descendents(label_name_id=label_name_id, name=label_name, max_depth=depth)


@router.get("/names", response_model=Dict[int, schemas.LabelName])
async def get_all_label_names(limit: int = 1000, offset: int = 0, atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    return atriumdb_sdk.get_all_label_names(limit=limit, offset=offset)
