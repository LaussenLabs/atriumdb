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

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'filtered-get-all'


def test_filtered_get_all():
    _test_for_both(DB_NAME, _test_filtered_get_all)


def _test_filtered_get_all(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # -------------------------------------------------------------------------
    # Setup: Create measures, devices, patients, and write data with known
    # time ranges so we can verify filtering.
    # -------------------------------------------------------------------------

    # Measures
    hr_id = sdk.insert_measure(measure_tag="HeartRate", freq=1.0, freq_units="Hz", units="BPM")
    rr_id = sdk.insert_measure(measure_tag="RespRate", freq=1.0, freq_units="Hz", units="BPM")
    spo2_id = sdk.insert_measure(measure_tag="SpO2", freq=0.5, freq_units="Hz", units="%")

    # Devices
    dev_a_id = sdk.insert_device(device_tag="DeviceA")
    dev_b_id = sdk.insert_device(device_tag="DeviceB")
    dev_c_id = sdk.insert_device(device_tag="DeviceC")  # No data will be written for this device

    # Patients
    patient_1_id = sdk.insert_patient(mrn="PAT-001")
    patient_2_id = sdk.insert_patient(mrn="PAT-002")

    # Map devices to patients with time ranges
    # DeviceA -> Patient 1: 0s to 100s
    # DeviceB -> Patient 2: 50s to 200s
    sdk.insert_device_patient_data([
        (dev_a_id, patient_1_id, 0, 100_000_000_000),          # 0s to 100s in nano
        (dev_b_id, patient_2_id, 50_000_000_000, 200_000_000_000),  # 50s to 200s in nano
    ])

    # Write data:
    # DeviceA + HeartRate: 0s to 60s
    values_a_hr = np.arange(60, dtype=np.float64)
    sdk.write_segment(hr_id, dev_a_id, values_a_hr, 0.0, freq=1.0, freq_units="Hz", time_units="s")

    # DeviceA + RespRate: 10s to 40s
    values_a_rr = np.arange(30, dtype=np.float64)
    sdk.write_segment(rr_id, dev_a_id, values_a_rr, 10.0, freq=1.0, freq_units="Hz", time_units="s")

    # DeviceB + HeartRate: 50s to 150s
    values_b_hr = np.arange(100, dtype=np.float64)
    sdk.write_segment(hr_id, dev_b_id, values_b_hr, 50.0, freq=1.0, freq_units="Hz", time_units="s")

    # DeviceB + SpO2: 80s to 180s
    values_b_spo2 = np.arange(50, dtype=np.float64)
    sdk.write_segment(spo2_id, dev_b_id, values_b_spo2, 80.0, freq=0.5, freq_units="Hz", time_units="s")

    # No data written for DeviceC or for DeviceA+SpO2 or DeviceB+RespRate

    # -------------------------------------------------------------------------
    # Test 1: No-arg behavior is unchanged
    # -------------------------------------------------------------------------
    all_measures = sdk.get_all_measures()
    assert set(all_measures.keys()) == {hr_id, rr_id, spo2_id}, \
        f"Expected all 3 measures, got {set(all_measures.keys())}"

    all_devices = sdk.get_all_devices()
    assert set(all_devices.keys()) == {dev_a_id, dev_b_id, dev_c_id}, \
        f"Expected all 3 devices, got {set(all_devices.keys())}"

    # -------------------------------------------------------------------------
    # Test 2: get_all_measures filtered by device_id
    # -------------------------------------------------------------------------
    # DeviceA has HeartRate and RespRate
    measures_dev_a = sdk.get_all_measures(device_id=dev_a_id)
    assert set(measures_dev_a.keys()) == {hr_id, rr_id}, \
        f"Expected HR and RR for DeviceA, got {set(measures_dev_a.keys())}"

    # DeviceB has HeartRate and SpO2
    measures_dev_b = sdk.get_all_measures(device_id=dev_b_id)
    assert set(measures_dev_b.keys()) == {hr_id, spo2_id}, \
        f"Expected HR and SpO2 for DeviceB, got {set(measures_dev_b.keys())}"

    # DeviceC has no data
    measures_dev_c = sdk.get_all_measures(device_id=dev_c_id)
    assert len(measures_dev_c) == 0, \
        f"Expected no measures for DeviceC, got {set(measures_dev_c.keys())}"

    # -------------------------------------------------------------------------
    # Test 3: get_all_measures filtered by device_tag
    # -------------------------------------------------------------------------
    measures_dev_a_tag = sdk.get_all_measures(device_tag="DeviceA")
    assert set(measures_dev_a_tag.keys()) == {hr_id, rr_id}

    # -------------------------------------------------------------------------
    # Test 4: get_all_measures filtered by device_id + time range
    # -------------------------------------------------------------------------
    # DeviceA + HeartRate: 0s-60s, RespRate: 10s-40s
    # Query 0s-5s should only find HeartRate
    measures_dev_a_early = sdk.get_all_measures(device_id=dev_a_id, start_time=0, end_time=5, time_units="s")
    assert set(measures_dev_a_early.keys()) == {hr_id}, \
        f"Expected only HR in 0-5s on DeviceA, got {set(measures_dev_a_early.keys())}"

    # Query 15s-35s should find both HeartRate and RespRate
    measures_dev_a_mid = sdk.get_all_measures(device_id=dev_a_id, start_time=15, end_time=35, time_units="s")
    assert set(measures_dev_a_mid.keys()) == {hr_id, rr_id}, \
        f"Expected HR and RR in 15-35s on DeviceA, got {set(measures_dev_a_mid.keys())}"

    # Query 65s-100s should find nothing (HeartRate ends at 60s, RespRate at 40s)
    measures_dev_a_late = sdk.get_all_measures(device_id=dev_a_id, start_time=65, end_time=100, time_units="s")
    assert len(measures_dev_a_late) == 0, \
        f"Expected no measures in 65-100s on DeviceA, got {set(measures_dev_a_late.keys())}"

    # -------------------------------------------------------------------------
    # Test 5: get_all_measures filtered by patient (mrn)
    # -------------------------------------------------------------------------
    # Patient 1 is on DeviceA which has HeartRate and RespRate
    measures_pat1 = sdk.get_all_measures(mrn="PAT-001")
    assert set(measures_pat1.keys()) == {hr_id, rr_id}, \
        f"Expected HR and RR for patient 1, got {set(measures_pat1.keys())}"

    # Patient 2 is on DeviceB which has HeartRate and SpO2
    measures_pat2 = sdk.get_all_measures(patient_id=patient_2_id)
    assert set(measures_pat2.keys()) == {hr_id, spo2_id}, \
        f"Expected HR and SpO2 for patient 2, got {set(measures_pat2.keys())}"

    # -------------------------------------------------------------------------
    # Test 6: get_all_measures filtered by patient + time range
    # -------------------------------------------------------------------------
    # Patient 2 is on DeviceB: HR 50-150s, SpO2 80-180s
    # Query 50-75s should find only HeartRate
    measures_pat2_early = sdk.get_all_measures(
        mrn="PAT-002", start_time=50, end_time=75, time_units="s")
    assert set(measures_pat2_early.keys()) == {hr_id}, \
        f"Expected only HR for patient 2 in 50-75s, got {set(measures_pat2_early.keys())}"

    # Query 85-145s should find both HR and SpO2
    measures_pat2_mid = sdk.get_all_measures(
        patient_id=patient_2_id, start_time=85, end_time=145, time_units="s")
    assert set(measures_pat2_mid.keys()) == {hr_id, spo2_id}, \
        f"Expected HR and SpO2 for patient 2 in 85-145s, got {set(measures_pat2_mid.keys())}"

    # -------------------------------------------------------------------------
    # Test 7: get_all_measures with nonexistent mrn returns empty
    # -------------------------------------------------------------------------
    measures_nonexistent = sdk.get_all_measures(mrn="DOES-NOT-EXIST")
    assert len(measures_nonexistent) == 0

    # -------------------------------------------------------------------------
    # Test 8: get_all_devices filtered by measure_id
    # -------------------------------------------------------------------------
    # HeartRate is on DeviceA and DeviceB
    devices_hr = sdk.get_all_devices(measure_id=hr_id)
    assert set(devices_hr.keys()) == {dev_a_id, dev_b_id}, \
        f"Expected DeviceA and DeviceB for HR, got {set(devices_hr.keys())}"

    # RespRate is only on DeviceA
    devices_rr = sdk.get_all_devices(measure_id=rr_id)
    assert set(devices_rr.keys()) == {dev_a_id}, \
        f"Expected only DeviceA for RR, got {set(devices_rr.keys())}"

    # SpO2 is only on DeviceB
    devices_spo2 = sdk.get_all_devices(measure_id=spo2_id)
    assert set(devices_spo2.keys()) == {dev_b_id}, \
        f"Expected only DeviceB for SpO2, got {set(devices_spo2.keys())}"

    # -------------------------------------------------------------------------
    # Test 9: get_all_devices filtered by measure_tag
    # -------------------------------------------------------------------------
    devices_hr_tag = sdk.get_all_devices(measure_tag="HeartRate", freq=1.0, freq_units="Hz", units="BPM")
    assert set(devices_hr_tag.keys()) == {dev_a_id, dev_b_id}

    # -------------------------------------------------------------------------
    # Test 10: get_all_devices filtered by measure_id + time range
    # -------------------------------------------------------------------------
    # HeartRate on DeviceA: 0-60s, on DeviceB: 50-150s
    # Query 0-45s should find only DeviceA
    devices_hr_early = sdk.get_all_devices(measure_id=hr_id, start_time=0, end_time=45, time_units="s")
    assert set(devices_hr_early.keys()) == {dev_a_id}, \
        f"Expected only DeviceA for HR in 0-45s, got {set(devices_hr_early.keys())}"

    # Query 55-145s should find both
    devices_hr_mid = sdk.get_all_devices(measure_id=hr_id, start_time=55, end_time=145, time_units="s")
    assert set(devices_hr_mid.keys()) == {dev_a_id, dev_b_id}, \
        f"Expected DeviceA and DeviceB for HR in 55-145s, got {set(devices_hr_mid.keys())}"

    # Query 155-200s should find neither (HR ends at 60s and 150s)
    devices_hr_late = sdk.get_all_devices(measure_id=hr_id, start_time=155, end_time=200, time_units="s")
    assert len(devices_hr_late) == 0, \
        f"Expected no devices for HR in 155-200s, got {set(devices_hr_late.keys())}"

    # -------------------------------------------------------------------------
    # Test 11: get_all_devices filtered by patient
    # -------------------------------------------------------------------------
    # Patient 1 is on DeviceA
    devices_pat1 = sdk.get_all_devices(patient_id=patient_1_id)
    assert set(devices_pat1.keys()) == {dev_a_id}, \
        f"Expected DeviceA for patient 1, got {set(devices_pat1.keys())}"

    # Patient 2 is on DeviceB
    devices_pat2 = sdk.get_all_devices(mrn="PAT-002")
    assert set(devices_pat2.keys()) == {dev_b_id}, \
        f"Expected DeviceB for patient 2, got {set(devices_pat2.keys())}"

    # -------------------------------------------------------------------------
    # Test 12: get_all_devices filtered by measure + patient
    # -------------------------------------------------------------------------
    # Patient 2 + HeartRate should return DeviceB
    devices_pat2_hr = sdk.get_all_devices(measure_id=hr_id, patient_id=patient_2_id)
    assert set(devices_pat2_hr.keys()) == {dev_b_id}, \
        f"Expected DeviceB for patient 2 + HR, got {set(devices_pat2_hr.keys())}"

    # Patient 1 + SpO2 should return nothing (DeviceA has no SpO2)
    devices_pat1_spo2 = sdk.get_all_devices(measure_id=spo2_id, patient_id=patient_1_id)
    assert len(devices_pat1_spo2) == 0, \
        f"Expected no devices for patient 1 + SpO2, got {set(devices_pat1_spo2.keys())}"

    # -------------------------------------------------------------------------
    # Test 13: get_all_measures/devices with only time range (no device/patient)
    # -------------------------------------------------------------------------
    # 0-5s: only DeviceA+HR has data
    measures_time_only = sdk.get_all_measures(start_time=0, end_time=5, time_units="s")
    assert hr_id in measures_time_only, "HR should have data in 0-5s"
    assert rr_id not in measures_time_only, "RR starts at 10s, should not be in 0-5s"

    # 155-200s: only DeviceB+SpO2 has data (ends at 180s)
    measures_late = sdk.get_all_measures(start_time=155, end_time=200, time_units="s")
    assert set(measures_late.keys()) == {spo2_id}, \
        f"Expected only SpO2 in 155-200s, got {set(measures_late.keys())}"

    # Devices in 155-200s: only DeviceB
    devices_late = sdk.get_all_devices(start_time=155, end_time=200, time_units="s")
    assert set(devices_late.keys()) == {dev_b_id}, \
        f"Expected only DeviceB in 155-200s, got {set(devices_late.keys())}"

    # -------------------------------------------------------------------------
    # Test 14: get_interval_array time_units parameter
    # -------------------------------------------------------------------------
    # DeviceA + HeartRate: 0s to 60s
    intervals_nano = sdk.get_interval_array(measure_id=hr_id, device_id=dev_a_id)
    assert intervals_nano.shape[0] > 0, "Should have intervals"

    # Same query using time_units="s"
    intervals_s = sdk.get_interval_array(
        measure_id=hr_id, device_id=dev_a_id, start=0, end=100, time_units="s")
    assert intervals_s.shape[0] > 0, "Should have intervals with time_units='s'"
    # The returned values should still be in nanoseconds
    assert intervals_s[0][0] >= 0
    assert intervals_s[0][0] < 1_000_000, "Start should be near 0ns, not 0s"

    # -------------------------------------------------------------------------
    # Test 15: time_units validation
    # -------------------------------------------------------------------------
    try:
        sdk.get_all_measures(device_id=dev_a_id, start_time=0, end_time=10)
        assert False, "Should have raised ValueError for missing time_units"
    except ValueError:
        pass  # Expected

    try:
        sdk.get_all_devices(measure_id=hr_id, start_time=0, end_time=10)
        assert False, "Should have raised ValueError for missing time_units"
    except ValueError:
        pass  # Expected

    try:
        sdk.get_interval_array(measure_id=hr_id, device_id=dev_a_id, start=0, end=10, time_units="invalid")
        assert False, "Should have raised ValueError for invalid time_units"
    except ValueError:
        pass  # Expected

    # -------------------------------------------------------------------------
    # Test 16: Verify return dict values match original format
    # -------------------------------------------------------------------------
    measures_filtered = sdk.get_all_measures(device_id=dev_a_id)
    for mid, minfo in measures_filtered.items():
        assert 'id' in minfo
        assert 'tag' in minfo
        assert 'freq_nhz' in minfo
        assert 'period_ns' in minfo
        assert 'unit' in minfo
        assert 'source_id' in minfo
        assert minfo['id'] == mid

    devices_filtered = sdk.get_all_devices(measure_id=hr_id)
    for did, dinfo in devices_filtered.items():
        assert 'id' in dinfo
        assert 'tag' in dinfo
        assert 'manufacturer' in dinfo
        assert 'bed_id' in dinfo
        assert 'source_id' in dinfo
        assert dinfo['id'] == did

    print("All filtered get_all tests passed!")
