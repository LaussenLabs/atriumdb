from atriumdb import AtriumSDK
import numpy as np

from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both

DB_NAME = "atrium-get-data-patient"


def test_insert_data_with_patient_mapping():
    _test_for_both(DB_NAME, _test_insert_data_with_patient_mapping)


def _test_insert_data_with_patient_mapping(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_tag = 'signal_1'
    freq_hz = 1
    period = 1 / freq_hz

    freq_nhz = int(freq_hz * (10 ** 9))
    period_ns = int(period * (10 ** 9))

    device_tag = 'dev_1'

    measure_id = sdk.insert_measure(measure_tag, freq_hz, freq_units="Hz")
    device_id = sdk.insert_device(device_tag)

    num_values = 1_000_000
    og_time_data = np.arange(num_values, dtype=np.int64) * period_ns
    og_value_data = np.sin(np.arange(num_values))

    sdk.write_data_easy(
        measure_id, device_id, og_time_data, og_value_data, freq_hz)

    patient_id = insert_random_patients(sdk, 1)[0]

    # Map the device to the patient
    start_time = int(og_time_data[0])
    end_time = int(og_time_data[-1] + period_ns)
    sdk.insert_device_patient_data([(device_id, patient_id, start_time, end_time)])

    _, patient_read_times, patient_read_values = sdk.get_data(measure_id, start_time, end_time, patient_id=patient_id)

    assert np.array_equal(og_time_data, patient_read_times)
    assert np.array_equal(og_value_data, patient_read_values)
