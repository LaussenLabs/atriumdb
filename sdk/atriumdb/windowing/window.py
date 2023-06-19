import numpy as np
import logging

from dataclasses import dataclass, field
from typing import Tuple

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
    device_id: str
    window_config: WindowConfig
    # a signal will be stored for each measure ID (measure_name, freq, uom). freq and uom can be None, meaning any value
    signals: dict[Tuple[str, float | None, str | None], Signal] = field(default_factory=dict)
    end_time: int = field(init=False)
