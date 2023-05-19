import numpy as np
from fastapi.testclient import TestClient

from atriumdb.atrium_sdk import AtriumSDK
from tests.generate_wfdb import get_records

from tests.mock_api.app import app
from tests.mock_api.sdk_dependency import get_sdk_instance
from tests.test_mit_bih import write_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'api_test'
MAX_RECORDS = 1


def test_api():
    _test_for_both(DB_NAME, _test_api)


def _test_api(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS)

    client = TestClient(app)
    client.app.dependency_overrides[get_sdk_instance] = lambda: sdk

    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="", api_test_client=client)
    assert_mit_bih_to_dataset_api(api_sdk, max_records=MAX_RECORDS)

    # response = client.get("/v1/sdk/blocks", params={
    #     "start_time": 0,
    #     "end_time": 1000,
    #     "measure_id": 1,
    #     "device_id": 123
    # })
    # assert response.status_code == 400


def assert_mit_bih_to_dataset_api(api_sdk, max_records=None):
    num_records = 0
    for record in get_records(dataset_name='mitdb'):
        if max_records and num_records >= max_records:
            return
        num_records += 1
        device_id = api_sdk.get_device_id(device_tag=record.record_name)
        assert device_id is not None
        freq_nano = record.fs * 1_000_000_000
        period_ns = int(10 ** 9 // record.fs)

        time_arr = np.arange(record.sig_len, dtype=np.int64) * period_ns

        if record.n_sig > 1:
            for i in range(len(record.sig_name)):
                measure_id = api_sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano,
                                                    units=record.units[i])
                assert measure_id is not None

                start_time = time_arr[0]
                end_time = time_arr[-1] + period_ns

                # Replace sdk.get_data_api call with TestClient
                headers, r_times, r_values = api_sdk.get_data_api(measure_id, start_time, end_time, device_id=device_id)

                assert np.array_equal(record.p_signal.T[i], r_values) and np.array_equal(time_arr, r_times)

        else:
            measure_id = api_sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano,
                                                units=record.units)
            assert measure_id is not None

            start_time = time_arr[0]
            end_time = time_arr[-1] + period_ns

            headers, r_times, r_values = api_sdk.get_data_api(measure_id, end_time, start_time, device_id=device_id)

            assert np.array_equal(record.p_signal, r_values) and np.array_equal(time_arr, r_times)


def mock_get_data_api(sdk, client: TestClient, measure_id, end_time, start_time, device_id):
    block_info_url = sdk.get_block_info_api_url(measure_id, start_time, end_time, device_id, None, None)
    block_info_response = client.get(block_info_url)
    block_info_list = block_info_response.json()
    if len(block_info_list) == 0:
        return [], np.array([], dtype=np.int64)
    block_requests = []
    for block_info in block_info_list:
        block_id = block_info['id']
        block_request_url = sdk.api_url + f"/v1/sdk/blocks/{block_id}"
        response = client.get(block_request_url)
        block_requests.append(response)
    encoded_bytes = np.concatenate(
        [np.frombuffer(response.content, dtype=np.uint8) for response in block_requests], axis=None)
    num_bytes_list = [row['num_bytes'] for row in block_info_list]
    headers, r_times, r_values = \
        sdk.decode_block_arr(encoded_bytes, num_bytes_list, start_time, end_time, True, True)
    return headers, r_times, r_values
