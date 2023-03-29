from typing import Optional

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Security, Response, Request

from atriumdb import AtriumSDK
from tests.mock_api.sdk_dependency import get_sdk_instance

sdk_router = APIRouter()


@sdk_router.get("/blocks")
async def get_blocks(
        start_time: int,
        end_time: int,
        measure_id: int,
        patient_id: Optional[int] = None,
        mrn: Optional[str] = None,
        device_id: Optional[int] = None,
        sdk: AtriumSDK = Depends(get_sdk_instance),
):
    # BASE VALIDATION
    if measure_id is None:
        raise HTTPException(status_code=400, detail="A measure_id must be specified")
    if sum(x is not None for x in (device_id, patient_id, mrn)) != 1:
        raise HTTPException(status_code=400, detail="Exactly one of device_id, patient_id, or mrn must be specified")
    if start_time > end_time:
        raise HTTPException(status_code=400, detail="The start time must be lower than the end time")

    if device_id is not None:
        device_info = sdk.get_device_info(device_id=device_id)

        # If the device_id doesn't exist in atriumdb raise an error
        if device_info is None:
            raise HTTPException(status_code=400, detail=f"Cannot find device_id={device_id} in AtriumDB")

    if patient_id is not None:
        patient_id_dict = sdk.get_patient_id_to_mrn_map([patient_id])
        if patient_id not in patient_id_dict:
            raise HTTPException(status_code=400, detail=f"Cannot find patient_id={patient_id} in AtriumDB.")

    if mrn is not None:
        mrn_patient_id_map = sdk.get_mrn_to_patient_id_map([int(mrn)])
        if mrn not in mrn_patient_id_map:
            raise HTTPException(status_code=400, detail=f"Cannot find mrn={mrn} in AtriumDB.")
        patient_id = mrn_patient_id_map[mrn]

    block_info = []
    block_id_list = sdk.sql_handler.select_blocks(
        measure_id, start_time_n=start_time, end_time_n=end_time,
        device_id=device_id, patient_id=patient_id)

    # Get indices of the block_id_list tuple
    id_idx = 0
    start_time_n_idx = 6
    end_time_n_idx = 7
    num_values_idx = 8
    num_bytes_idx = 5

    for row in block_id_list:
        block_info.append({'id': row[id_idx],
                           'start_time_n': row[start_time_n_idx],
                           'end_time_n': row[end_time_n_idx],
                           'num_values': row[num_values_idx],
                           'num_bytes': row[num_bytes_idx]})

    return block_info


@sdk_router.get("/blocks/{block_id}")
async def get_block(
        block_id: int,
        sdk: AtriumSDK = Depends(get_sdk_instance),
):
    # Get block_info
    block_info = sdk.sql_handler.select_block(block_id=block_id)

    if block_info is None:
        raise HTTPException(status_code=404, detail=f"Cannot find block_id={block_id} in AtriumDB.")

    # Get indices of the block_info tuple
    measure_id_idx = 1
    device_id_idx = 2
    file_id_idx = 3
    start_byte_idx = 4
    num_bytes_idx = 5

    file_id_list = [block_info[file_id_idx]]
    filename = sdk.get_filename_dict(file_id_list)[block_info[file_id_idx]]

    if not Path(filename).is_file():
        filename = sdk.file_api.to_abs_path(filename, block_info[measure_id_idx], block_info[device_id_idx])

    with open(filename, 'rb') as file:
        file.seek(block_info[start_byte_idx])
        return Response(file.read(block_info[num_bytes_idx]))
