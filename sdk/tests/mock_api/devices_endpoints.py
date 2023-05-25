from typing import Optional

from fastapi import APIRouter, HTTPException, Depends

from atriumdb import AtriumSDK
from tests.mock_api.sdk_dependency import get_sdk_instance

devices_router = APIRouter()


@devices_router.get("/")
async def search_devices(
        device_tag: Optional[str] = None,
        device_name: Optional[str] = None,
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):

    if device_tag is None and device_name is None:
        res = atriumdb_sdk.get_all_devices()
    else:
        res = atriumdb_sdk.search_devices(tag_match=device_tag, name_match=device_name)

    if res is None:
        raise HTTPException(status_code=404, detail="No devices Found")
    return res


@devices_router.get("/{device_id}")
async def get_device_info(
        device_id: int,
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):

    res = atriumdb_sdk.get_device_info(device_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"No device found for device_id {device_id}")
    return res
