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

from typing import Optional, List, Tuple
from tests.mock_api.sdk_dependency import get_sdk_instance
from fastapi import APIRouter, Depends, HTTPException
from atriumdb import AtriumSDK
import tests.mock_api.schemas as schemas

router = APIRouter()


@router.get("/{id}", response_model=schemas.Patient)
async def get_patient_info(
        id: str,
        time: Optional[int] = None,
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    # check to make sure the user is using proper identifier format
    if "id|" not in id and "mrn|" not in id:
        raise HTTPException(status_code=400, detail="Patient id or mrn malformed. Must be of the structure 'id|12345' if searching by patient id or 'mrn|1234567' if searching by mrn")

    # split on pipe character to see if the prefix is "id" or "mrn" and query accordingly
    split = id.split('|')
    if split[0] == 'mrn':
        res = atriumdb_sdk.get_patient_info(mrn=int(split[1]), time=time)
    elif split[0] == 'id':
        res = atriumdb_sdk.get_patient_info(patient_id=int(split[1]), time=time)
    else:
        raise HTTPException(status_code=400, detail="Request string malformed please use the form 'id|12345' or 'mrn|1234567'")

    if res is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return res


@router.get("/{id}/history", response_model=List[Tuple])
async def get_patient_history(
        id: str,
        field: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    # check to make sure the user is using proper identifier format
    if "id|" not in id and "mrn|" not in id:
        raise HTTPException(status_code=400, detail="Patient id or mrn malformed. Must be of the structure 'id|12345' if searching by patient id or 'mrn|1234567' if searching by mrn")

    # split on pipe character to see if the prefix is "id" or "mrn" and query accordingly
    split = id.split('|')
    if split[0] == 'mrn':
        res = atriumdb_sdk.get_patient_history(mrn=int(split[1]), field=field, start_time=start_time, end_time=end_time)
    elif split[0] == 'id':
        res = atriumdb_sdk.get_patient_history(patient_id=int(split[1]), field=field, start_time=start_time, end_time=end_time)
    else:
        raise HTTPException(status_code=400, detail="Request string malformed please use the form 'id|12345' or 'mrn|1234567'")

    if res is None:
        raise HTTPException(status_code=404, detail=f"No patient history found for id={id}, field={field}, start_time={start_time}, end_time={end_time}")
    return res