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
