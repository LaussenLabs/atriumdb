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

def _ingest_data_tsc(to_sdk, measure_id, device_id, headers, times, values):
    try:
        if all([h.scale_m == headers[0].scale_m and
                h.scale_b == headers[0].scale_b and
                h.freq_nhz == headers[0].freq_nhz for h in headers]):

            to_sdk.write_data(measure_id, device_id, times, values, headers[0].freq_nhz, int(times[0]),
                              raw_time_type=headers[0].t_raw_type, raw_value_type=headers[0].v_raw_type,
                              encoded_time_type=headers[0].t_encoded_type, encoded_value_type=headers[0].v_encoded_type,
                              scale_m=headers[0].scale_m, scale_b=headers[0].scale_b, interval_index_mode="fast")
        else:
            val_index = 0
            for h in headers:
                try:
                    to_sdk.write_data(measure_id, device_id, times, values, h.freq_nhz, int(times[0]),
                                      raw_time_type=h.t_raw_type, raw_value_type=h.v_raw_type,
                                      encoded_time_type=h.t_encoded_type,
                                      encoded_value_type=h.v_encoded_type,
                                      scale_m=h.scale_m, scale_b=h.scale_b,
                                      interval_index_mode="fast")
                except IndexError:
                    continue
                val_index += h.num_vals
    except Exception as e:
        print(e)
        raise e
