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

import json
from typing import Dict, Optional, Union
from pathlib import Path, PurePath


def create_json_metadata_dict(measure_tag: str, freq: int, unit: str, device_tag: str, start_nano: int, end_nano: int,
                              scale_m: Optional[float] = None, scale_b: Optional[float] = None,
                              patient_id: Optional[int] = None) -> Dict[str, Union[str, float, int]]:
    metadata = {
        "measure_tag": measure_tag,
        "freq": freq,
        "unit": unit,
        "device_tag": device_tag,
        "start_nano": start_nano,
        "end_nano": end_nano,
    }
    if scale_m is not None:
        metadata["scale_m"] = scale_m
    if scale_b is not None:
        metadata["scale_b"] = scale_b
    if patient_id is not None:
        metadata["patient_id"] = patient_id

    return metadata


def export_json_metadata(directory: Union[str, PurePath], file_metadata: Dict[str, Dict[str, Union[str, float, int]]]):
    directory = Path(directory)
    with open(directory / "file_list.json", 'w') as f:
        json.dump(file_metadata, f, indent=2)


def import_json_metadata(directory: Union[str, PurePath]) -> Dict[str, Dict[str, Union[str, float, int]]]:
    directory = Path(directory)
    with open(directory / "file_list.json", 'r') as f:
        file_metadata = json.load(f)
    return file_metadata
