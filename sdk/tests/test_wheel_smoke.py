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
Checks that an installed atriumdb loads its native library from the package and round-trips data through it.

Runs under pytest, or directly as a script (python test_wheel_smoke.py) against an installed wheel.
"""

import tempfile
from pathlib import Path

import numpy as np

import atriumdb
from atriumdb import AtriumSDK
from atriumdb.helpers.block_constants import COMPRESSION_TYPES

NUM_SAMPLES = 500_000  # Several blocks, so multi-threaded encoding and decoding is exercised.
FREQ_HZ = 1000
PERIOD_NS = 10 ** 9 // FREQ_HZ
START_NS = 1_700_000_000 * 10 ** 9


def test_library_ships_in_package():
    library_dir = Path(atriumdb.__file__).parent / "bin"
    assert list(library_dir.glob("libTSC.*")), f"no native library in {library_dir}"


def test_round_trip_every_compression_type():
    for name in ("ZSTD", "LZ4", "LZ4HC"):
        _round_trip(name)


def _round_trip(compression_name):
    times = START_NS + np.arange(NUM_SAMPLES, dtype=np.int64) * PERIOD_NS
    values = np.round(1000 * np.sin(np.arange(NUM_SAMPLES) / 50.0)).astype(np.float64)

    with tempfile.TemporaryDirectory() as dataset_location:
        AtriumSDK.create_dataset(dataset_location=dataset_location, database_type="sqlite")
        sdk = AtriumSDK(dataset_location=dataset_location, num_threads=4)

        sdk.block.v_compression = COMPRESSION_TYPES[compression_name]
        sdk.block.v_compression_level = 0 if compression_name == "LZ4" else 5

        measure_id = sdk.insert_measure(measure_tag="smoke", freq=FREQ_HZ, freq_units="Hz", units="mV")
        device_id = sdk.insert_device(device_tag="smoke")
        sdk.write_data_easy(measure_id, device_id, times, values, FREQ_HZ, time_units="ns", freq_units="Hz")

        _, read_times, read_values = sdk.get_data(
            measure_id, START_NS, START_NS + NUM_SAMPLES * PERIOD_NS, device_id=device_id)

    assert np.array_equal(read_times, times), compression_name
    assert np.allclose(read_values, values), compression_name


if __name__ == "__main__":
    test_library_ships_in_package()
    test_round_trip_every_compression_type()
    print("atriumdb native library OK")
