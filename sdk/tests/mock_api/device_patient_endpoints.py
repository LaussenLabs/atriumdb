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

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from atriumdb import AtriumSDK
from tests.mock_api.sdk_dependency import get_sdk_instance

router = APIRouter()


@router.get("", response_model=List[List[Optional[int]]])
async def get_device_patient_data(
        device_id: List[int] = Query(default=[]),
        patient_id: List[int] = Query(default=[]),
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timestamp: Optional[int] = None,
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    if timestamp is not None and (start_time is not None or end_time is not None):
        raise HTTPException(
            status_code=400,
            detail="timestamp and start_time/end_time are mutually exclusive.")

    device_id_list = device_id if device_id else None
    patient_id_list = patient_id if patient_id else None

    if timestamp is not None:
        results = atriumdb_sdk.sql_handler.select_device_patient_encounters(
            timestamp=timestamp,
            device_id_list=device_id_list,
            patient_id_list=patient_id_list,
        )
    else:
        results = atriumdb_sdk.sql_handler.select_device_patients(
            device_id_list=device_id_list,
            patient_id_list=patient_id_list,
            start_time=start_time,
            end_time=end_time,
        )

    return [list(row) for row in results]