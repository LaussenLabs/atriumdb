import pytest
from pathlib import Path
import os

from atriumdb.transfer.adb.patients import create_patient_id_map, write_dict_to_csv


def test_direct_mapping():
    patient_id_list = [1, 2, 3]
    result = create_patient_id_map(patient_id_list, False)
    assert result == {1: 1, 2: 2, 3: 3}, "Direct mapping failed"


def test_deidentify_true():
    patient_id_list = [1, 2, 3]
    result = create_patient_id_map(patient_id_list, True)
    assert all(key != value for key, value in result.items()), "De-identification failed"
    assert len(set(result.values())) == len(patient_id_list), "Generated IDs are not unique"


def test_deidentify_with_new_file(tmp_path):
    patient_id_list = [1, 2, 3]
    file_path = tmp_path / "deid_map.csv"
    result = create_patient_id_map(patient_id_list, file_path, overwrite=True)
    assert os.path.exists(file_path), "File was not created"
    assert all(key != value for key, value in result.items()), "De-identification failed"


def test_deidentify_with_existing_file(tmp_path):
    patient_id_list = [1, 2, 3]
    file_path = tmp_path / "deid_map.csv"
    pre_existing_map = {1: 10001, 2: 10002, 3: 10003}
    write_dict_to_csv(pre_existing_map, file_path)

    result = create_patient_id_map(patient_id_list, file_path, overwrite=False)
    assert result == pre_existing_map, "Existing map was not used"


def test_deidentify_with_invalid_existing_file(tmp_path):
    patient_id_list = [1, 2, 3]
    file_path = tmp_path / "deid_map.csv"
    invalid_map = {1: 10001, 2: 10002}  # Missing patient_id 3
    write_dict_to_csv(invalid_map, file_path)

    with pytest.raises(ValueError):
        create_patient_id_map(patient_id_list, file_path, overwrite=False)
