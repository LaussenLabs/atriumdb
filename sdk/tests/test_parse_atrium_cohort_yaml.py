from atriumdb.transfer.cohort.cohort_yaml import parse_atrium_cohort_yaml
from pathlib import Path

YAML_FILENAME = Path(__file__).parent / 'example_data' / 'example_cohort.yaml'


def test_parse_atrium_cohort_yaml():
    filename = str(YAML_FILENAME)
    expected_output = {
        'measures': ['heart_rate', 'blood_pressure', 'oxygen_saturation', 'respiratory_rate'],
        'measure_ids': [1, 2, 3],
        'devices': ['1_101-A', '1_101-B ', '1_102-A ', 'Ventilator'],
        'device_ids': None,
        'patient_ids': [20001, 20002, 20003, 20004, 45409],
        'mrns': None,
        'start_epoch_s': 1615463200.0,
        'end_epoch_s': None,
    }
    output = parse_atrium_cohort_yaml(filename)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"
