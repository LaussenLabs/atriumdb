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
from atriumdb import AtriumSDK
import threading
import uvicorn
from tests.testing_framework import _test_for_both
from tests.mock_api.app import app
from tests.mock_api.sdk_dependency import get_sdk_instance
import pytest

DB_NAME = 'test_db_patient_history'


def test_patient_history():
    # test local mode
    _test_for_both(DB_NAME, _test_patient_history)

    def start_server():
        uvicorn.run(app)

    # start server in daemon thread so it exits when complete
    api_thread = threading.Thread(target=start_server, daemon=True)
    api_thread.start()

    # test api mode
    _test_for_both(DB_NAME, _test_patient_history_api)


def _test_patient_history(db_type, dataset_location, connection_params):

    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # insert patient and pull their info using current time to make sure it's correct
    patient_id = sdk.insert_patient(mrn=1234567, gender='M', first_name="Sterling", middle_name="Malory", last_name="Archer",
                                    dob=283901400000000, weight=83.461, weight_units='kg', height=188, height_units='cm')
    patient_info = sdk.get_patient_info(mrn=1234567, time=time.time_ns())

    # make sure height and weight are correct
    assert patient_info['height'] == 188
    assert patient_info['weight'] == 83.461
    assert patient_info['height_units'] == 'cm'
    assert patient_info['weight_units'] == 'kg'

    # make sure it also works using patient_id
    patient_info = sdk.get_patient_info(patient_id=1, time=time.time_ns())
    # make sure height and weight are correct
    assert patient_info['height'] == 188
    assert patient_info['weight'] == 83.461
    assert patient_info['height_units'] == 'cm'
    assert patient_info['weight_units'] == 'kg'

    # make the time before the patient was inserted to make sure no height and weight exists
    patient_info = sdk.get_patient_info(mrn=1234567, time=time.time_ns()-1_000_000_000_000)
    assert patient_info['height'] is None
    assert patient_info['weight'] is None
    assert patient_info['height_units'] is None
    assert patient_info['weight_units'] is None
    assert patient_info['height_time'] is None
    assert patient_info['weight_time'] is None

    # make sure patient history table was also updated
    patient_height_history = sdk.get_patient_history(mrn=1234567, field='height')
    assert len(patient_height_history) == 1
    assert patient_height_history[0][1] == 1  # make sure patient id is correct
    assert patient_height_history[0][2] == 'height'
    assert patient_height_history[0][3] == 188  # make sure patient height is correct
    assert patient_height_history[0][4] == 'cm'  # make sure height units are correct

    patient_weight_history = sdk.get_patient_history(patient_id=1, field='weight')
    assert len(patient_weight_history) == 1
    assert patient_weight_history[0][1] == 1  # make sure patient id is correct
    assert patient_weight_history[0][2] == 'weight'
    assert patient_weight_history[0][3] == 83.461  # make sure patient weight is correct
    assert patient_weight_history[0][4] == 'kg'  # make sure weight units are correct

    # insert some height and weight to the patient history table
    sdk.insert_patient_history(field='weight', value=81.4, units='kg', time=1707945246, time_units='s', patient_id=1)
    sdk.insert_patient_history(field='weight', value=83.2, units='kg', time=1707945346000000000, patient_id=1)
    sdk.insert_patient_history(field='height', value=180, units='cm', time=1707945446, time_units='s', patient_id=1)
    sdk.insert_patient_history(field='weight', value=83, units='kg', time=1707945546, time_units='s', patient_id=1)
    sdk.insert_patient_history(field='height', value=185, units='cm', time=1707945646, time_units='s', mrn=1234567)

    # the most recent height and weight should be pulled and it should be the original we inputted when adding patient
    patient_info = sdk.get_patient_info(mrn=1234567, time=time.time_ns())
    assert patient_info['height'] == 188
    assert patient_info['weight'] == 83.461

    # now lets go back in time and pull the patient and make sure we are getting the correct height and weight
    patient_info = sdk.get_patient_info(mrn=1234567, time=1707945546, time_units='s')
    assert patient_info['height'] == 180
    assert patient_info['weight'] == 83

    # now lets go do a time where the height will be none but there will be a weight
    patient_info = sdk.get_patient_info(patient_id=1, time=1707945346000000000)
    assert patient_info['height'] is None
    assert patient_info['weight'] == 83.2

    # get all patient history for height and confirm it is in the correct order and the info is correct
    patient_history = sdk.get_patient_history(patient_id=1, field='height')
    assert len(patient_history) == 3

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'height'
    assert patient_history[0][3] == 180
    assert patient_history[0][4] == 'cm'
    assert patient_history[0][5] == 1707945446000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'height'
    assert patient_history[1][3] == 185
    assert patient_history[1][4] == 'cm'
    assert patient_history[1][5] == 1707945646000000000

    assert patient_history[2][1] == 1
    assert patient_history[2][2] == 'height'
    assert patient_history[2][3] == 188
    assert patient_history[2][4] == 'cm'

    # get patient history with start and end time
    patient_history = sdk.get_patient_history(patient_id=1, field='weight', start_time=1707945346000000000,
                                              end_time=1708104586074435800)
    assert len(patient_history) == 2

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'weight'
    assert patient_history[0][3] == 83.2
    assert patient_history[0][4] == 'kg'
    assert patient_history[0][5] == 1707945346000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'weight'
    assert patient_history[1][3] == 83
    assert patient_history[1][4] == 'kg'
    assert patient_history[1][5] == 1707945546000000000

    # get patient history without an end time
    patient_history = sdk.get_patient_history(patient_id=1, field='weight', start_time=1707945346000000000)
    assert len(patient_history) == 3

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'weight'
    assert patient_history[0][3] == 83.2
    assert patient_history[0][4] == 'kg'
    assert patient_history[0][5] == 1707945346000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'weight'
    assert patient_history[1][3] == 83
    assert patient_history[1][4] == 'kg'
    assert patient_history[1][5] == 1707945546000000000

    assert patient_history[2][1] == 1
    assert patient_history[2][2] == 'weight'
    assert patient_history[2][3] == 83.461
    assert patient_history[2][4] == 'kg'

    # get patient history using mrn and without a start time
    patient_history = sdk.get_patient_history(mrn=1234567, field='height', end_time=1708104586074435800)
    assert len(patient_history) == 2

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'height'
    assert patient_history[0][3] == 180
    assert patient_history[0][4] == 'cm'
    assert patient_history[0][5] == 1707945446000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'height'
    assert patient_history[1][3] == 185
    assert patient_history[1][4] == 'cm'
    assert patient_history[1][5] == 1707945646000000000


def _test_patient_history_api(db_type, dataset_location, connection_params):

    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # set up the local sdk instance for the api to use
    app.dependency_overrides[get_sdk_instance] = lambda: sdk

    # set up remote mode sdk to connect to the api
    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="http://127.0.0.1:8000", validate_token=False)
    # change the sdk token expiry so the test can work
    api_sdk.token_expiry = time.time() + 1_000_000

    # insert patient and pull their info using current time to make sure it's correct
    patient_id = sdk.insert_patient(mrn=1234567, gender='M', first_name="Sterling", middle_name="Malory", last_name="Archer",
                                    dob=283901400000000, weight=83.461, weight_units='kg', height=188, height_units='cm')
    patient_info = api_sdk.get_patient_info(mrn=1234567, time=time.time_ns())

    # make sure height and weight are correct
    assert patient_info['height'] == 188
    assert patient_info['weight'] == 83.461
    assert patient_info['height_units'] == 'cm'
    assert patient_info['weight_units'] == 'kg'

    # make sure it also works using patient_id
    patient_info = api_sdk.get_patient_info(patient_id=1, time=time.time_ns())
    # make sure height and weight are correct
    assert patient_info['height'] == 188
    assert patient_info['weight'] == 83.461
    assert patient_info['height_units'] == 'cm'
    assert patient_info['weight_units'] == 'kg'

    # make the time before the patient was inserted to make sure no height and weight exists
    patient_info = api_sdk.get_patient_info(mrn=1234567, time=time.time_ns()-1_000_000_000_000)
    assert patient_info['height'] is None
    assert patient_info['weight'] is None
    assert patient_info['height_units'] is None
    assert patient_info['weight_units'] is None
    assert patient_info['height_time'] is None
    assert patient_info['weight_time'] is None

    # make sure patient history table was also updated
    patient_height_history = api_sdk.get_patient_history(mrn=1234567, field='height')
    assert len(patient_height_history) == 1
    assert patient_height_history[0][1] == 1  # make sure patient id is correct
    assert patient_height_history[0][2] == 'height'
    assert patient_height_history[0][3] == 188  # make sure patient height is correct
    assert patient_height_history[0][4] == 'cm'  # make sure height units are correct

    patient_weight_history = api_sdk.get_patient_history(patient_id=1, field='weight')
    assert len(patient_weight_history) == 1
    assert patient_weight_history[0][1] == 1  # make sure patient id is correct
    assert patient_weight_history[0][2] == 'weight'
    assert patient_weight_history[0][3] == 83.461  # make sure patient weight is correct
    assert patient_weight_history[0][4] == 'kg'  # make sure weight units are correct

    # insert some height and weight to the patient history table
    sdk.insert_patient_history(field='weight', value=81.4, units='kg', time=1707945246, time_units='s', patient_id=1)
    sdk.insert_patient_history(field='weight', value=83.2, units='kg', time=1707945346000000000, patient_id=1)
    sdk.insert_patient_history(field='height', value=180, units='cm', time=1707945446, time_units='s', patient_id=1)
    sdk.insert_patient_history(field='weight', value=83, units='kg', time=1707945546, time_units='s', patient_id=1)
    sdk.insert_patient_history(field='height', value=185, units='cm', time=1707945646, time_units='s', mrn=1234567)

    # the most recent height and weight should be pulled and it should be the original we inputted when adding patient
    patient_info = api_sdk.get_patient_info(mrn=1234567, time=time.time_ns())
    assert patient_info['height'] == 188
    assert patient_info['weight'] == 83.461

    # now lets go back in time and pull the patient and make sure we are getting the correct height and weight
    patient_info = api_sdk.get_patient_info(mrn=1234567, time=1707945546, time_units='s')
    assert patient_info['height'] == 180
    assert patient_info['weight'] == 83

    # now lets go do a time where the height will be none but there will be a weight
    patient_info = api_sdk.get_patient_info(patient_id=1, time=1707945346000000000)
    assert patient_info['height'] is None
    assert patient_info['weight'] == 83.2

    # get all patient history for height and confirm it is in the correct order and the info is correct
    patient_history = api_sdk.get_patient_history(patient_id=1, field='height')
    assert len(patient_history) == 3

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'height'
    assert patient_history[0][3] == 180
    assert patient_history[0][4] == 'cm'
    assert patient_history[0][5] == 1707945446000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'height'
    assert patient_history[1][3] == 185
    assert patient_history[1][4] == 'cm'
    assert patient_history[1][5] == 1707945646000000000

    assert patient_history[2][1] == 1
    assert patient_history[2][2] == 'height'
    assert patient_history[2][3] == 188
    assert patient_history[2][4] == 'cm'

    # get patient history with start and end time
    patient_history = api_sdk.get_patient_history(patient_id=1, field='weight', start_time=1707945346000000000,
                                                  end_time=1708104586074435800)
    assert len(patient_history) == 2

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'weight'
    assert patient_history[0][3] == 83.2
    assert patient_history[0][4] == 'kg'
    assert patient_history[0][5] == 1707945346000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'weight'
    assert patient_history[1][3] == 83
    assert patient_history[1][4] == 'kg'
    assert patient_history[1][5] == 1707945546000000000

    # get patient history without an end time
    patient_history = api_sdk.get_patient_history(patient_id=1, field='weight', start_time=1707945346000000000)
    assert len(patient_history) == 3

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'weight'
    assert patient_history[0][3] == 83.2
    assert patient_history[0][4] == 'kg'
    assert patient_history[0][5] == 1707945346000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'weight'
    assert patient_history[1][3] == 83
    assert patient_history[1][4] == 'kg'
    assert patient_history[1][5] == 1707945546000000000

    assert patient_history[2][1] == 1
    assert patient_history[2][2] == 'weight'
    assert patient_history[2][3] == 83.461
    assert patient_history[2][4] == 'kg'

    # get patient history using mrn and without a start time
    patient_history = api_sdk.get_patient_history(mrn=1234567, field='height', end_time=1708104586074435800)
    assert len(patient_history) == 2

    assert patient_history[0][1] == 1
    assert patient_history[0][2] == 'height'
    assert patient_history[0][3] == 180
    assert patient_history[0][4] == 'cm'
    assert patient_history[0][5] == 1707945446000000000

    assert patient_history[1][1] == 1
    assert patient_history[1][2] == 'height'
    assert patient_history[1][3] == 185
    assert patient_history[1][4] == 'cm'
    assert patient_history[1][5] == 1707945646000000000

    # malformed request should raise error
    with pytest.raises(ValueError):
        _ = api_sdk._request("GET", f"/patients/i|{patient_id}/history", params={'field': 'height'})

    # close api connection
    api_sdk.close()
