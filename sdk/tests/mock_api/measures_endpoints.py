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

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

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
