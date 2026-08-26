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
    """One window of data emitted by ``AtriumSDK.get_iterator``.

    :ivar dict signals: One entry per measure, keyed by ``(tag, freq_hz, units)``
        -- note that the key does NOT include the value type, so a numeric and a
        string measure sharing a tag/freq/units would collide. Each entry is::

            {'times':          (N,) int64   the GRID timestamps, not observation times
             'values':         (N,) float64, or (N,) int64 for string CODE channels
             'expected_count': int          N, the number of grid cells
             'actual_count':   int          cells that are not the unknown sentinel
             'measure_id':     int}

        ``N`` is per measure (``window_duration // that measure's raster
        period``), so a mixed-rate window has channels of different lengths.

        **Sentinels.** A cell whose value is not known -- a gap, an empty
        ``sparse`` cell, the region of a ``state`` before its first observed
        transition, or any cell of a trailing partial window that falls outside
        the definition range -- carries ``NaN`` on a numeric channel and
        :data:`~atriumdb.string_dictionary.UNKNOWN_STRING_CODE` (``-1``, which
        decodes to ``"<unknown>"``) on a string code channel. Real dictionary
        codes are always ``>= 0``, so ``values >= 0`` is an exact known-mask for
        string channels; a numeric channel has no such mask, because a genuinely
        written NaN reading is indistinguishable from an unfilled cell.

        **``actual_count`` measures fill coverage, not observation density.** A
        carried-forward stale value counts as present, and a genuinely written
        NaN counts as absent. For observation times or counts, read the raw
        stream with ``AtriumSDK.get_data`` / ``get_string_data`` over the same
        range.

        **String channels hold codes, not text.** Decode on demand with
        :meth:`decode_string_signal`. An ``event`` measure rendered with the
        default ``presence``/``count`` rule holds occupancy floats instead and
        cannot be decoded -- see :func:`assert_decodable_string_signal`.

    :ivar int start_time: Window start in epoch nanoseconds.
    :ivar int end_time: Exclusive logical end in epoch nanoseconds. This is always
        ``start_time + window_duration``; a trailing partial window may have data
        only through an earlier time.
    :ivar int device_id: Source device, or ``None`` for a patient source.
    :ivar int patient_id: Source patient, or ``None`` for a device source.
    :ivar np.ndarray label_time_series: ``(num_label_sets, row_size)`` int8 0/1
        per-sample label channel, on the FASTEST measure's grid, or ``None``.
    :ivar np.ndarray label: Window-level label per label set, obtained by
        thresholding ``label_time_series`` at ``label_threshold``, or ``None``.
    :ivar dict patient_info: Patient demographics for this window (plus any
        requested ``patient_history_fields``, resolved as of the window start).
    """

    signals: dict
    start_time: int
    end_time: int
    device_id: int
    patient_id: int
    label_time_series: np.ndarray
    label: np.ndarray
    patient_info: dict

    def decode_string_signal(self, sdk, measure_key, unknown_value=None):
        """Decode a string signal's int64 codes in this window to strings.

        String measures rasterize their dictionary CODES into the window, not
        decoded strings; this accessor decodes them on
        demand via the measure's :class:`MeasureStringDictionary`. ``measure_key``
        is the ``(tag, freq_hz, units)`` tuple keying :attr:`signals`. The
        reserved unknown sentinel decodes to ``unknown_value`` (``"<unknown>"``
        by default) instead of raising.

        Only a *code* channel can be decoded. A channel rendered with the
        ``presence`` / ``count`` fill rules -- which is what an ``event`` measure
        gets by default -- holds occupancy **floats**, not dictionary codes, so
        decoding it would silently fabricate a plausible-looking but entirely
        wrong sequence of clinical strings. Such a channel raises
        :class:`ValueError` here; see :func:`assert_decodable_string_signal`."""
        # Imported lazily to avoid a module import cycle at load time.
        from atriumdb.string_dictionary import decode_window_codes
        signal = self.signals[measure_key]
        assert_decodable_string_signal(signal, measure_key)
        return decode_window_codes(sdk, signal['measure_id'], signal['values'],
                                   unknown_value=unknown_value)


def assert_decodable_string_signal(signal, measure_key=None):
    """Raise unless ``signal['values']`` really holds int64 dictionary codes.

    The window's value dtype is the exact discriminator: rasterized string codes
    are ``int64`` (with ``UNKNOWN_STRING_CODE`` for censored cells), while every
    numeric channel -- including the ``presence`` / ``count`` rasterization of a
    *string* ``event`` measure -- is ``float64``.

    Blindly ``astype(int64)``-ing a presence channel turns 0/1 occupancy into
    dictionary codes 0/1 and returns real vocabulary words for cells where
    nothing happened. That is silent fabrication of clinical values through a
    documented API, so it is refused instead.

    There is no "right" decoding for a presence/count channel: it is an
    occupancy count over a grid cell, deliberately lossy about *which* value
    occurred (a cell may contain several distinct events). To read the strings
    themselves use :meth:`AtriumSDK.get_string_data` /
    :meth:`AtriumSDK.get_event_intervals` on the same time range, which return
    the raw event stream rather than a rasterized grid.
    """
    values = np.asarray(signal['values'])
    if np.issubdtype(values.dtype, np.integer):
        return
    where = f" for signal {measure_key}" if measure_key is not None else ""
    raise ValueError(
        f"Cannot decode string codes{where}: the window's values have dtype "
        f"{values.dtype}, not int64 dictionary codes. This channel is a numeric "
        f"rasterization -- an 'event' measure filled with 'presence'/'count' holds "
        f"occupancy floats, and a numeric measure holds measurements; neither carries "
        f"decodable codes. Decoding it would fabricate vocabulary strings from those "
        f"numbers. To read the underlying strings use AtriumSDK.get_string_data() or "
        f"AtriumSDK.get_event_intervals() over the same time range; to get a decodable "
        f"code channel from a string measure, render it with a code-preserving fill "
        f"rule (e.g. fill_overrides={{measure_id: 'sparse'}} on a 'sample'/'state' kind).")
