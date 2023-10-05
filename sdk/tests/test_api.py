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
from fastapi.testclient import TestClient

from atriumdb.adb_functions import sort_data
from atriumdb.atrium_sdk import AtriumSDK
from tests.generate_wfdb import get_records

from tests.mock_api.app import app
from tests.mock_api.sdk_dependency import get_sdk_instance
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'api_test'
MAX_RECORDS = 1
SEED = 42


def test_api():
    _test_for_both(DB_NAME, _test_api)


def _test_api(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    client = TestClient(app)
    client.app.dependency_overrides[get_sdk_instance] = lambda: sdk

    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="", api_test_client=client)
    assert_mit_bih_to_dataset(api_sdk, max_records=MAX_RECORDS, seed=SEED)


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
    r_times, r_values, headers = sdk.block.decode_blocks(encoded_bytes, num_bytes_list, analog=True,
                                                         time_type=1)
    r_times, r_values = sort_data(r_times, r_values, headers, start_time, end_time)

    return headers, r_times, r_values
