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

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.transfer.adb.dataset import transfer_data
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = 'atrium-transfer-time-ranges'


def test_transfer_block_across_ranges():
    _test_for_both(DB_NAME, _test_transfer_block_across_ranges)


def _test_transfer_block_across_ranges(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    # Readings every 15 minutes with a 3 hour outage, written as one block.
    s, t0 = 10 ** 9, 1_790_812_800 * 10 ** 9
    measure_id = sdk_1.insert_measure("readings", freq=1 / 900, freq_units="Hz", units="1")
    device_id = sdk_1.insert_device("bed")
    times = t0 + np.arange(48, dtype=np.int64) * 900 * s
    times = times[(times < t0 + 4 * 3600 * s) | (times >= t0 + 7 * 3600 * s)]
    sdk_1.write_time_value_pairs(measure_id, device_id, times, np.arange(times.size, dtype=np.float64),
                                 freq=1 / 900, freq_units="Hz")

    # Overlapping ranges, one ending a minute after a reading, and a range after the outage.
    ranges = [{'start': int(times[0]), 'end': int(times[4]) + 60 * s},
              {'start': int(times[2]), 'end': int(times[8])},
              {'start': t0 + 7 * 3600 * s, 'end': int(times[-1]) + 1}]
    definition = DatasetDefinition(measures=[("readings", 1 / 900, "1")], device_ids={device_id: ranges})
    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False)

    dest_measure_id = sdk_2.get_measure_id("readings", freq=1 / 900, freq_units="Hz", units="1")
    _, dest_times, _ = sdk_2.get_data(dest_measure_id, 0, 2 ** 62, device_id=sdk_2.get_device_id("bed"))
    assert np.array_equal(dest_times, times[(times < times[8]) | (times >= t0 + 7 * 3600 * s)])
