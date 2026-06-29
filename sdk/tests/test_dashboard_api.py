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
import time
import threading
from pathlib import Path

import numpy as np
import requests
import uvicorn

from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.dashboard.measure_queries import query_measure_total_hours
from atriumdb.dashboard.schemas import (
    AdmissionDateRange, AgeBand, CohortDefinitionRequest,
    DemographicCohort, MrnCohort,
)
from tests.mock_api.app import app
from tests.mock_api.sdk_dependency import get_sdk_instance

DB_NAME = 'dashboard_api_test'
SQLITE_DATASET_PATH = Path(__file__).parent / "test_datasets" / f"sqlite_{DB_NAME}"

DB_NAME_HOURS = 'dashboard_api_hours_test'
SQLITE_DATASET_PATH_HOURS = Path(__file__).parent / "test_datasets" / f"sqlite_{DB_NAME_HOURS}"
HOURS_API_PORT = 8124


def test_api_cohorts():
    def start_server():
        uvicorn.run(app, port=8123)

    api_thread = threading.Thread(target=start_server, daemon=True)
    api_thread.start()

    shutil.rmtree(SQLITE_DATASET_PATH, ignore_errors=True)
    _test_api_cohorts('sqlite', SQLITE_DATASET_PATH, None)


def _test_api_cohorts(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    app.dependency_overrides[get_sdk_instance] = lambda: sdk

    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="http://127.0.0.1:8123", validate_token=False)
    api_sdk.token_expiry = time.time() + 1_000_000

    # --- set up location infrastructure (institution → unit → bed) ---
    institution_id = sdk.sql_handler.insert_institution("Test Hospital")
    unit_id = sdk.sql_handler.insert_unit(institution_id, "ICU", "icu")
    bed_id = sdk.sql_handler.insert_bed(unit_id, "Bed 1")

    ONE_YEAR_NS = 365 * 24 * 3600 * 1_000_000_000
    admit_start_ns = 1_600_000_000_000_000_000
    admit_end_ns   = 1_700_000_000_000_000_000
    inside_ns      = 1_650_000_000_000_000_000  # within the admission window
    outside_ns     = 1_500_000_000_000_000_000  # before the admission window

    # patient A: male, 25 years old at admission, has in-window encounter in ICU
    pid_a = sdk.insert_patient(mrn="MRN001", gender="M", dob=inside_ns - 25 * ONE_YEAR_NS)
    # patient B: female, 35 years old at admission, has in-window encounter in ICU
    pid_b = sdk.insert_patient(mrn="MRN002", gender="F", dob=inside_ns - 35 * ONE_YEAR_NS)
    # patient C: male, but encounter is outside the admission window
    pid_c = sdk.insert_patient(mrn="MRN003", gender="M", dob=inside_ns - 25 * ONE_YEAR_NS)

    sdk.insert_encounter(patient_id=pid_a, bed_id=bed_id, start_time=inside_ns,
                         end_time=inside_ns + ONE_YEAR_NS, visit_number="V001")
    sdk.insert_encounter(patient_id=pid_b, bed_id=bed_id,
                         start_time=inside_ns + ONE_YEAR_NS // 2,
                         end_time=inside_ns + ONE_YEAR_NS, visit_number="V002")
    sdk.insert_encounter(patient_id=pid_c, bed_id=bed_id, start_time=outside_ns,
                         end_time=outside_ns + ONE_YEAR_NS, visit_number="V003")

    date_range = AdmissionDateRange(start=admit_start_ns, end=admit_end_ns)

    print("Testing 1A: MRN cohort endpoint...")

    request_1a = CohortDefinitionRequest(
        type="mrn",
        admission_date_range=date_range,
        cohorts=[MrnCohort(id="cohort_1a", mrn_list=["MRN001", "MRN002", "MRN003", "MRN999"])],
    )

    # MRN001 and MRN002 have in-window encounters; MRN003 is outside the window; MRN999 does not exist
    local_result = sdk.dashboard_resolve_cohort(request_1a, request_id="test-1a-local")
    assert local_result.request_id == "test-1a-local"
    assert len(local_result.cohorts) == 1
    assert local_result.cohorts[0].id == "cohort_1a"
    assert set(local_result.cohorts[0].mrn_list) == {"MRN001", "MRN002"}

    api_result = api_sdk.dashboard_resolve_cohort(request_1a, request_id="test-1a-api")
    assert set(api_result.cohorts[0].mrn_list) == {"MRN001", "MRN002"}

    print("Testing 1B: demographic cohort — location filter...")

    request_1b_loc = CohortDefinitionRequest(
        type="demographic",
        admission_date_range=date_range,
        cohorts=[DemographicCohort(id="cohort_1b_loc", location=["ICU"])],
    )

    # both in-window patients are in ICU; MRN003 is outside the window
    local_result = sdk.dashboard_resolve_cohort(request_1b_loc)
    assert set(local_result.cohorts[0].mrn_list) == {"MRN001", "MRN002"}

    api_result = api_sdk.dashboard_resolve_cohort(request_1b_loc)
    assert set(api_result.cohorts[0].mrn_list) == {"MRN001", "MRN002"}

    print("Testing 1B: demographic cohort — sex filter...")

    request_1b_sex = CohortDefinitionRequest(
        type="demographic",
        admission_date_range=date_range,
        cohorts=[DemographicCohort(id="cohort_1b_sex", location=["ICU"], sex=["M"])],
    )

    local_result = sdk.dashboard_resolve_cohort(request_1b_sex)
    assert set(local_result.cohorts[0].mrn_list) == {"MRN001"}

    api_result = api_sdk.dashboard_resolve_cohort(request_1b_sex)
    assert set(api_result.cohorts[0].mrn_list) == {"MRN001"}

    print("Testing 1B: demographic cohort — age filter...")

    # patient A is 25 years old at admission → falls in [20, 30]; patient B is 35 → does not
    request_1b_age = CohortDefinitionRequest(
        type="demographic",
        admission_date_range=date_range,
        cohorts=[DemographicCohort(
            id="cohort_1b_age",
            location=["ICU"],
            age=[AgeBand(start_ns=20 * ONE_YEAR_NS, end_ns=30 * ONE_YEAR_NS)],
        )],
    )

    local_result = sdk.dashboard_resolve_cohort(request_1b_age)
    assert set(local_result.cohorts[0].mrn_list) == {"MRN001"}

    api_result = api_sdk.dashboard_resolve_cohort(request_1b_age)
    assert set(api_result.cohorts[0].mrn_list) == {"MRN001"}

    print("Testing 1B: demographic cohort — multiple cohorts in one request...")

    request_multi = CohortDefinitionRequest(
        type="demographic",
        admission_date_range=date_range,
        cohorts=[
            DemographicCohort(id="male_icu",  location=["ICU"], sex=["M"]),
            DemographicCohort(id="female_icu", location=["ICU"], sex=["F"]),
        ],
    )

    local_result = sdk.dashboard_resolve_cohort(request_multi)
    assert len(local_result.cohorts) == 2
    cohorts_by_id = {c.id: set(c.mrn_list) for c in local_result.cohorts}
    assert cohorts_by_id["male_icu"] == {"MRN001"}
    assert cohorts_by_id["female_icu"] == {"MRN002"}

    api_result = api_sdk.dashboard_resolve_cohort(request_multi)
    cohorts_by_id = {c.id: set(c.mrn_list) for c in api_result.cohorts}
    assert cohorts_by_id["male_icu"] == {"MRN001"}
    assert cohorts_by_id["female_icu"] == {"MRN002"}

    api_sdk.close()


def test_api_measure_total_hours():
    def start_server():
        uvicorn.run(app, port=HOURS_API_PORT)

    api_thread = threading.Thread(target=start_server, daemon=True)
    api_thread.start()

    shutil.rmtree(SQLITE_DATASET_PATH_HOURS, ignore_errors=True)
    _test_api_measure_total_hours('sqlite', SQLITE_DATASET_PATH_HOURS, None)


def _test_api_measure_total_hours(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    app.dependency_overrides[get_sdk_instance] = lambda: sdk

    # --- create measures and devices ---
    hr_id = sdk.insert_measure(measure_tag="HR", freq=1, freq_units="Hz", unit="BPM")
    spo2_id = sdk.insert_measure(measure_tag="SpO2", freq=1, freq_units="Hz", unit="%")
    dev1_id = sdk.insert_device(device_tag="monitor_1")
    dev2_id = sdk.insert_device(device_tag="monitor_2")

    # Use a fixed base timestamp (seconds) to keep numbers predictable.
    base_s = 1_700_000_000

    # HR on device 1: 7200 samples at 1 Hz → 2 hours
    t_hr_d1 = np.arange(base_s, base_s + 7200, dtype=np.int64)
    v_hr_d1 = np.zeros(7200, dtype=np.float64)
    sdk.write_data_easy(hr_id, dev1_id, t_hr_d1, v_hr_d1, freq=1, freq_units="Hz", time_units="s")

    # HR on device 2: 3600 samples at 1 Hz → 1 hour
    t_hr_d2 = np.arange(base_s, base_s + 3600, dtype=np.int64)
    v_hr_d2 = np.zeros(3600, dtype=np.float64)
    sdk.write_data_easy(hr_id, dev2_id, t_hr_d2, v_hr_d2, freq=1, freq_units="Hz", time_units="s")

    # SpO2 on device 1: 3600 samples at 1 Hz → 1 hour
    t_spo2_d1 = np.arange(base_s, base_s + 3600, dtype=np.int64)
    v_spo2_d1 = np.zeros(3600, dtype=np.float64)
    sdk.write_data_easy(spo2_id, dev1_id, t_spo2_d1, v_spo2_d1, freq=1, freq_units="Hz", time_units="s")

    print("Testing measure_total_hours: local helper...")

    local_result = query_measure_total_hours(sdk)
    assert len(local_result) == 2

    by_tag = {r["measure_tag"]: r for r in local_result}

    assert set(by_tag["HR"].keys()) == {"measure_id", "measure_tag", "freq_nhz", "units",
                                        "num_devices", "total_ns", "total_hours"}
    assert abs(by_tag["HR"]["total_hours"] - 3.0) < 1e-6
    assert by_tag["HR"]["num_devices"] == 2
    assert abs(by_tag["SpO2"]["total_hours"] - 1.0) < 1e-6
    assert by_tag["SpO2"]["num_devices"] == 1

    print("Testing measure_total_hours: API endpoint GET /measures/hours...")

    # Give the server a moment to be ready.
    time.sleep(0.5)

    resp = requests.get(f"http://127.0.0.1:{HOURS_API_PORT}/measures/hours", timeout=10)
    assert resp.status_code == 200

    api_result = resp.json()
    assert len(api_result) == 2

    by_tag_api = {r["measure_tag"]: r for r in api_result}

    assert abs(by_tag_api["HR"]["total_hours"] - 3.0) < 1e-6
    assert by_tag_api["HR"]["num_devices"] == 2
    assert abs(by_tag_api["SpO2"]["total_hours"] - 1.0) < 1e-6
    assert by_tag_api["SpO2"]["num_devices"] == 1

    print("All measure_total_hours tests passed.")
