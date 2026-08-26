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
import shutil
from pathlib import Path

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.transfer.adb.dataset import transfer_data
from atriumdb.windowing.verify_definition import verify_definition
from tests.test_transfer_info import insert_random_patients

DB_NAME = "atrium-orphan-device-patient"

ORPHAN_DEVICE_ID = 109


# The device_patient table can contain rows referencing devices that don't exist in the device
# table (SQLite doesn't enforce the foreign key on normal connections; API sources can be
# inconsistent the same way). MariaDB enforces the FK with ON DELETE CASCADE, so the orphan row
# can't be created there and this test runs against SQLite only.
def test_transfer_with_orphan_device_patient_mapping():
    dataset_location = Path(__file__).parent / "test_datasets" / f"sqlite_{DB_NAME}"
    dest_location = Path(__file__).parent / "test_datasets" / f"sqlite_{DB_NAME}_2"
    shutil.rmtree(dataset_location, ignore_errors=True)
    shutil.rmtree(dest_location, ignore_errors=True)

    sdk_1 = AtriumSDK.create_dataset(dataset_location=dataset_location, database_type="sqlite")
    sdk_2 = AtriumSDK.create_dataset(dataset_location=dest_location, database_type="sqlite")

    measure_tag = 'signal_1'
    freq_hz = 1
    period_ns = 10 ** 9

    measure_id = sdk_1.insert_measure(measure_tag, freq_hz, freq_units="Hz")
    device_id = sdk_1.insert_device('dev_1')

    num_values = 10_000
    time_data = np.arange(num_values, dtype=np.int64) * period_ns
    value_data = np.sin(np.arange(num_values))

    sdk_1.write_data_easy(measure_id, device_id, time_data, value_data, freq_hz)

    patient_id = insert_random_patients(sdk_1, 1)[0]

    start_time = int(time_data[0])
    end_time = int(time_data[-1] + period_ns)
    mid_time = (start_time + end_time) // 2

    # Simulate an orphaned mapping: a device_patient row whose device has no row in the device
    # table, overlapping the same patient's time range.
    assert sdk_1.get_device_info(ORPHAN_DEVICE_ID) is None
    sdk_1.sql_handler.insert_device_patients([(ORPHAN_DEVICE_ID, patient_id, start_time, mid_time)])
    sdk_1.insert_device_patient_data([(device_id, patient_id, start_time, end_time)])

    definition = DatasetDefinition(measures=[measure_tag], patient_ids={patient_id: "all"})

    # Validation should drop the orphaned device with a warning, keeping the real mapping.
    with pytest.warns(UserWarning, match=f"device id {ORPHAN_DEVICE_ID}"):
        _, _, mapped_sources = verify_definition(definition, sdk_1)

    mapped_device_ids = {dev_id for dev_id, _ in mapped_sources['device_patient_tuples'].keys()}
    assert device_id in mapped_device_ids
    assert ORPHAN_DEVICE_ID not in mapped_device_ids

    # The full transfer should complete and write the exported definition file.
    definition = DatasetDefinition(measures=[measure_tag], patient_ids={patient_id: "all"})
    transfer_data(sdk_1, sdk_2, definition, deidentify=False, include_labels=False)

    assert (Path(dest_location) / "meta" / "definition.yaml").is_file()

    _, read_times, read_values = sdk_2.get_data(measure_id, start_time, end_time, device_id=device_id)
    assert np.array_equal(time_data, read_times)
    assert np.array_equal(value_data, read_values)
