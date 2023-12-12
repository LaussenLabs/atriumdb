def transfer_measures(from_sdk, to_sdk, measure_id_list=None):
    from_measures = from_sdk.get_all_measures()

    measure_map = {}
    for from_measure_id, measure_info in from_measures.items():
        if measure_id_list is None or from_measure_id in measure_id_list:
            measure_tag = measure_info['tag']
            freq = measure_info['freq_nhz']
            units = measure_info['unit']
            measure_name = measure_info['name']
            to_measure_id = to_sdk.insert_measure(
                measure_tag=measure_tag, freq=freq, units=units, measure_name=measure_name, measure_id=from_measure_id)

            measure_map[from_measure_id] = to_measure_id

    return measure_map
