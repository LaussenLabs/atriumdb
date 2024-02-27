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

import numpy as np
import logging

from dataclasses import dataclass, field
from typing import Tuple, Union

from atriumdb.windowing.window_config import WindowConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Signal:
    data: np.ndarray(shape=(), dtype=float) = None  # numpy array of raw measurement values
    times: np.ndarray(shape=(), dtype=int) = None  # numpy array of the timestamps of received values
    total_count: int = 0  # counter of received samples for this signal
    complete: bool = False  # tracker of completion
    # below will be added once first sample for this signal is received
    expected_count: int = None
    sample_rate: float = None
    source_id: str = None
    measurement_type: str = None
    unit_of_measure: str = None


@dataclass(slots=True)
class CommonWindowFormat:
    start_time: int
    end_time: int
    device_id: str
    window_config: WindowConfig
    # a signal will be stored for each measure ID (measure_name, freq, uom). freq and uom can be None, meaning any value
    signals: dict[Tuple[str, Union[float, int, None], Union[str, None]], Signal] = field(default_factory=dict)


@dataclass
class Window:
    signals: dict
    start_time: int
    device_id: int
    patient_id: int
    label_time_series: np.ndarray
    label: np.ndarray
    patient_info: dict
