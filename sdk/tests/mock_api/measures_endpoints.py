from typing import Optional, Dict
from fastapi import APIRouter, Depends, Security, HTTPException

from atriumdb import AtriumSDK
from tests.mock_api.sdk_dependency import get_sdk_instance

measures_router = APIRouter()


@measures_router.get("/")
async def search_measures(
        measure_tag: Optional[str] = None,
        measure_name: Optional[str] = None,
        unit: Optional[str] = None,
        freq: Optional[int | float] = None,
        freq_units: Optional[str] = None,
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):

    if measure_tag is None and measure_name is None and unit is None and freq is None:
        return atriumdb_sdk.get_all_measures()
    else:
        res = atriumdb_sdk.search_measures(
            tag_match=measure_tag, freq=freq, unit=unit, name_match=measure_name, freq_units=freq_units)

    if res is None:
        raise HTTPException(status_code=404, detail="No Measures Found")
    return res


@measures_router.get("/{measure_id}")
async def get_measure_info(
        measure_id: int,
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):

    res = atriumdb_sdk.get_measure_info(measure_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"No Measures Found for {measure_id}")
    return res
