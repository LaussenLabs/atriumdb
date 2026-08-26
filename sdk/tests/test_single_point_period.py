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
"""
write_time_value_pairs on a SINGLE time-value pair uses the measure's declared period.

One timestamp has no interval to measure, so automatic detection has nothing to work
with: it used to warn and substitute a hard-coded ``1_000_000_000``. That constant is
only "one second" if the timestamps were already in nanoseconds -- with
``time_units="s"`` it was converted a second time and stored as ``1e18`` ns. The
measure's own ``period_ns`` is already in hand at that point and is guaranteed real
(``insert_measure`` rejects ``freq <= 0`` / ``period <= 0``), so declared metadata is
used instead. Live event ingest is one point per message, so this is the hot path.

The multi-sample path is deliberately untouched -- detection's statistics are doing real
work there -- and there is a test below that fails if it ever starts being overridden.

SQLite only (no backend difference is at stake):
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_single_point_period.py -q
"""
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from atriumdb import AtriumSDK
from atriumdb.block_wrapper import get_period_ns_from_header

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC


@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_1pt_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def stored_period_ns(sdk, measure_id, device_id, start_n, end_n, analog=True):
    """The period actually written with the block(s) covering [start_n, end_n).

    ``analog=False`` for a string measure requests its raw dictionary codes.
    """
    headers, _times, _values = sdk.get_data(
        measure_id, start_n, end_n, device_id=device_id, time_units="ns", analog=analog)
    assert len(headers) >= 1, "no block was written"
    return get_period_ns_from_header(headers[0])


# --------------------------------------------------------------------------- #
# The fix
# --------------------------------------------------------------------------- #
def test_single_point_write_stores_the_measures_declared_period(sdk):
    """A 0.5 Hz measure -> a 2 s period, not the fabricated 1 s."""
    measure_id = sdk.insert_measure(measure_tag="nibp", freq=0.5, freq_units="Hz", units="mmHg",
                                    signal_kind="sample")
    device_id = sdk.insert_device(device_tag="dev")
    assert sdk.get_measure_info(measure_id)['period_ns'] == 2 * SEC

    sdk.write_time_value_pairs(measure_id, device_id,
                               np.array([BASE], dtype=np.int64),
                               np.array([120.0], dtype=np.float64))

    assert stored_period_ns(sdk, measure_id, device_id, BASE, BASE + 10 * SEC) == 2 * SEC


def test_single_point_write_no_longer_warns(sdk):
    """The 'Cannot detect period from fewer than 2 timestamps' warning fired on every
    single-point write -- i.e. on every message of a live event stream."""
    measure_id = sdk.insert_measure(measure_tag="hr", freq=1, freq_units="Hz", units="bpm",
                                    signal_kind="sample")
    device_id = sdk.insert_device(device_tag="dev")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdk.write_time_value_pairs(measure_id, device_id,
                                   np.array([BASE], dtype=np.int64),
                                   np.array([88.0], dtype=np.float64))


def test_single_point_write_in_seconds_is_not_scaled_twice(sdk):
    """The sharpest edge of the old behaviour: detection returned 1_000_000_000 (already
    a nanosecond count) and write_time_value_pairs then converted it FROM seconds, so a
    time_units="s" single-point write stored a period of 1e18 ns -- 31.7 years."""
    measure_id = sdk.insert_measure(measure_tag="alarm", freq=1, freq_units="Hz", units="bpm",
                                    signal_kind="sample")
    device_id = sdk.insert_device(device_tag="dev")

    start_s = 1_600_000_000.0
    sdk.write_time_value_pairs(measure_id, device_id,
                               np.array([start_s], dtype=np.float64),
                               np.array([1.0], dtype=np.float64),
                               time_units="s")

    period = stored_period_ns(sdk, measure_id, device_id, int(start_s * SEC),
                              int(start_s * SEC) + 10 * SEC)
    assert period == SEC
    assert period != 10 ** 18


def test_single_point_string_write_uses_declared_period(sdk):
    """The motivating workload: one event string per message."""
    measure_id = sdk.insert_measure(measure_tag="alarm_text", freq=0.25, freq_units="Hz",
                                    units="string", signal_kind="event", value_type="string")
    device_id = sdk.insert_device(device_tag="dev")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdk.write_time_value_pairs(measure_id, device_id,
                                   np.array([BASE], dtype=np.int64),
                                   np.array(["ASYSTOLE"], dtype=object))

    assert stored_period_ns(sdk, measure_id, device_id, BASE, BASE + 60 * SEC,
                            analog=False) == 4 * SEC
    times, values = sdk.get_string_data(measure_id, start_time_n=BASE, end_time_n=BASE + 60 * SEC,
                                        device_id=device_id)
    assert list(values) == ["ASYSTOLE"]


def test_buffered_single_point_flush_uses_declared_period(sdk):
    """The deferred-detection site: a buffered write leaves period/freq unset until
    flush, and a flush can come down to a single point too."""
    measure_id = sdk.insert_measure(measure_tag="vent_mode", freq=0.2, freq_units="Hz",
                                    units="mmHg", signal_kind="sample")
    device_id = sdk.insert_device(device_tag="dev")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with sdk.write_buffer():
            sdk.write_time_value_pairs(measure_id, device_id,
                                       np.array([BASE], dtype=np.int64),
                                       np.array([3.0], dtype=np.float64))

    assert stored_period_ns(sdk, measure_id, device_id, BASE, BASE + 60 * SEC) == 5 * SEC


# --------------------------------------------------------------------------- #
# What must NOT change
# --------------------------------------------------------------------------- #
def test_multi_point_write_still_detects_the_period_from_the_data(sdk):
    """Two or more timestamps: detection wins, the declared period must not override it.
    Measure declared at 1 Hz, data arriving every 5 s -> 5 s is stored."""
    measure_id = sdk.insert_measure(measure_tag="hr", freq=1, freq_units="Hz", units="bpm",
                                    signal_kind="sample")
    device_id = sdk.insert_device(device_tag="dev")

    times = BASE + np.arange(4, dtype=np.int64) * (5 * SEC)
    sdk.write_time_value_pairs(measure_id, device_id, times,
                               np.arange(4, dtype=np.float64))

    assert stored_period_ns(sdk, measure_id, device_id, BASE, BASE + 60 * SEC) == 5 * SEC


def test_explicit_period_still_wins_on_a_single_point(sdk):
    measure_id = sdk.insert_measure(measure_tag="hr", freq=1, freq_units="Hz", units="bpm",
                                    signal_kind="sample")
    device_id = sdk.insert_device(device_tag="dev")

    sdk.write_time_value_pairs(measure_id, device_id,
                               np.array([BASE], dtype=np.int64),
                               np.array([70.0], dtype=np.float64),
                               period=7 * SEC, time_units="ns")

    assert stored_period_ns(sdk, measure_id, device_id, BASE, BASE + 60 * SEC) == 7 * SEC


def test_explicit_freq_still_wins_on_a_single_point(sdk):
    measure_id = sdk.insert_measure(measure_tag="hr", freq=1, freq_units="Hz", units="bpm",
                                    signal_kind="sample")
    device_id = sdk.insert_device(device_tag="dev")

    sdk.write_time_value_pairs(measure_id, device_id,
                               np.array([BASE], dtype=np.int64),
                               np.array([70.0], dtype=np.float64),
                               freq=4, freq_units="Hz")

    assert stored_period_ns(sdk, measure_id, device_id, BASE, BASE + 60 * SEC) == SEC // 4


# --------------------------------------------------------------------------- #
# The helper itself
# --------------------------------------------------------------------------- #
def test_nominal_period_ns_reads_the_declared_period(sdk):
    measure_id = sdk.insert_measure(measure_tag="hr", period=250, time_units="ms", units="bpm")
    assert sdk._nominal_period_ns(measure_id) == 250_000_000


def test_nominal_period_ns_is_none_for_an_unknown_measure(sdk):
    """The caller keeps its previous fallback rather than inventing a period."""
    assert sdk._nominal_period_ns(999_999) is None


def test_nominal_period_ns_is_none_when_the_period_floors_to_zero(sdk):
    """Defensive: a hand-edited row whose frequency is too high for an integer
    nanosecond period. insert_measure cannot produce one."""
    assert sdk._nominal_period_ns(1, measure_info={'period_ns': None, 'freq_nhz': 10 ** 19}) is None
    assert sdk._nominal_period_ns(1, measure_info={'period_ns': 0, 'freq_nhz': 10 ** 9}) is None
