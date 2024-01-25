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
import time
import threading
import uvicorn
from atriumdb.atrium_sdk import AtriumSDK
from tests.mock_api.app import app
from tests.mock_api.sdk_dependency import get_sdk_instance
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'api_test'
MAX_RECORDS = 1
SEED = 42


def test_api():
    def start_server():
        uvicorn.run(app)

    # start server in daemon thread so it exits when complete
    websocket_connect_thread = threading.Thread(target=start_server, daemon=True)
    websocket_connect_thread.start()

    _test_for_both(DB_NAME, _test_api)


def _test_api(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    app.dependency_overrides[get_sdk_instance] = lambda: sdk

    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="http://127.0.0.1:8000", validate_token=False)
    # change the sdk token expiry so the test can work
    api_sdk.token_expiry = time.time() + 1_000_000

    assert_mit_bih_to_dataset(api_sdk, max_records=MAX_RECORDS, seed=SEED)
    # close api connection
    api_sdk.close()

