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
