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

def transfer_label_sets(src_sdk, dest_sdk, label_set_id_list=None):
    src_labels = src_sdk.get_all_label_names()
    label_set_map = {}
    for src_label_set_id, label_set_info in src_labels.items():
        if label_set_id_list is None or src_label_set_id in label_set_id_list:
            label_set_name = label_set_info['name']
            label_parent_id = label_set_info['parent_id']
            dest_label_set_id = dest_sdk.insert_label_name(
                label_set_name, label_name_id=src_label_set_id, parent=label_parent_id)
            label_set_map[src_label_set_id] = dest_label_set_id

    return label_set_map
