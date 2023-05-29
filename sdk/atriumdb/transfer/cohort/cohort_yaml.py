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

import yaml


def parse_atrium_cohort_yaml(filename):
    with open(filename, 'r') as stream:
        data = yaml.safe_load(stream)

    result = {
        'measures': data.get('measures') if data.get('measures') else None,
        'measure_ids': data.get('measure_ids') if data.get('measure_ids') else None,
        'devices': data.get('devices') if data.get('devices') else None,
        'device_ids': data.get('device_ids') if data.get('device_ids') else None,
        'patient_ids': data.get('patient_ids') if data.get('patient_ids') else None,
        'mrns': data.get('mrns') if data.get('mrns') else None,
    }

    start_epoch = data.get('start_epoch_s')
    if start_epoch and start_epoch != 'null' and start_epoch != '~':
        result['start_epoch_s'] = float(start_epoch)
    else:
        result['start_epoch_s'] = None

    end_epoch = data.get('end_epoch_s')
    if end_epoch and end_epoch != 'null' and end_epoch != '~':
        result['end_epoch_s'] = float(end_epoch)
    else:
        result['end_epoch_s'] = None

    return result
