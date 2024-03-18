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

measure_tag_match_rule_options = ["all", "best"]


def transfer_measures(from_sdk, to_sdk, measure_id_list=None, measure_tag_list=None, measure_tag_match_rule=None):
    if measure_id_list is not None and measure_tag_list is not None:
        raise ValueError("Only one of measure_id_list or measure_tag_list should be provided")

    measure_tag_match_rule = "all" if measure_tag_match_rule is None else measure_tag_match_rule
    assert measure_tag_match_rule in measure_tag_match_rule_options, \
        f"measure_tag_match_rule must be one of {measure_tag_match_rule_options}"

    from_measures = from_sdk.get_all_measures()
    measure_map = {}

    if measure_id_list:
        for from_measure_id in measure_id_list:
            if from_measure_id in from_measures:
                measure_map[from_measure_id] = _transfer_measure(to_sdk, from_measures[from_measure_id])

    elif measure_tag_list:
        for tag in measure_tag_list:
            if measure_tag_match_rule == "best":
                matching_ids = from_sdk.get_measure_id_list_from_tag(tag, approx=True)
                best_id = matching_ids[0] if matching_ids else None
                if best_id in from_measures:
                    measure_map[best_id] = _transfer_measure(to_sdk, from_measures[best_id])
            elif measure_tag_match_rule == "all":
                for from_measure_id, measure_info in from_measures.items():
                    if measure_info['tag'] == tag:
                        measure_map[from_measure_id] = _transfer_measure(to_sdk, measure_info)

    return measure_map


def _transfer_measure(to_sdk, measure_info):
    measure_tag = measure_info['tag']
    freq = measure_info['freq_nhz']
    units = measure_info['unit']
    measure_name = measure_info['name']
    from_measure_id = measure_info['id']

    # Check if measure_id already exists
    check_measure_info = to_sdk.get_measure_info(from_measure_id)
    if check_measure_info is not None:
        # if its the same measure, return the id without inserting
        if check_measure_info['tag'] == measure_tag and \
                check_measure_info['freq_nhz'] == freq and \
                check_measure_info['unit'] == units:
            return from_measure_id
        else:
            # The measure_id is taken but its a different measure so ask for a new id when inserting
            from_measure_id = None

    to_measure_id = to_sdk.insert_measure(
        measure_tag=measure_tag, freq=freq, units=units, measure_name=measure_name, measure_id=from_measure_id)
    return to_measure_id
