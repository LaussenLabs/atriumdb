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
import random
import time
from multiprocessing import freeze_support

from gen_helper import generate_allowable_freq_dict, get_all_allowable_freq

MAX_START_TIME_NS = 3265179796000000000  # June 20, 2073
MAX_FREQ_NHZ = 10000000000000  # 10 kHz

value_data_type_dict = {0: np.float32, 1: np.float64, 2: np.int8, 3: np.int16,
                        4: np.int32, 5: np.int64}
value_struct_char_dict = {0: 'f', 1: 'd', 2: 'b', 3: 'h', 4: 'i', 5: 'q'}
value_py_type_dict = {0: float, 1: float, 2: int, 3: int, 4: int, 5: int}
value_metadata_data_type = np.dtype('<u4')


def generate_random_sdk_compliant_data(num_messages, samples_per_message=None, start_time=None):
    start_time_ns = random.randint(0, MAX_START_TIME_NS) if start_time is None else start_time
    value_type_i = random.randint(0, len(value_py_type_dict) - 1)

    samples_per_message = random.randint(1, 1000) if \
        samples_per_message is None else samples_per_message

    freq_nhz = random.choice(get_all_allowable_freq(samples_per_message, 1000000, MAX_FREQ_NHZ, 1000000))

    # message_period_ns = round(float((10 ** 18) * samples_per_message) / float(freq_nhz))
    # print("freq_nhz", freq_nhz, "samples_per_message", samples_per_message, "message_period_ns", message_period_ns)

    value_data = generate_random_value_data(list(value_data_type_dict.values())[value_type_i],
                                            samples_per_message, num_messages)

    nom_times, server_times = generate_random_time_data(num_messages, samples_per_message, freq_nhz, start_time_ns,
                                                        start_time_ns)

    return value_data, nom_times, int(samples_per_message), int(freq_nhz)


def generate_random_value_data(data_type, samples_per_message, num_messages):
    start = random.randint(0, 1000000)
    end = random.randint(start + 1, start + 100000)
    noise = 0.1 * np.random.randn(num_messages * samples_per_message)
    values = noise + (0.8 * np.sin(
        np.linspace(start, end, num=num_messages * samples_per_message)))

    if not np.issubdtype(data_type, np.floating):
        values = values * np.iinfo(data_type).max
    values = values.astype(data_type)

    if samples_per_message == 1:
        return values
    else:
        null_offsets = np.zeros(num_messages, dtype=value_metadata_data_type)
        num_samples_arr = np.zeros(num_messages, dtype=value_metadata_data_type) + samples_per_message
        return values.reshape((num_messages, samples_per_message)), num_samples_arr, null_offsets


def generate_random_time_data(num_messages, samples_per_message, sample_freq_nhz, start_time_nominal,
                              start_time_server):
    message_period = (samples_per_message * (10 ** 18)) // sample_freq_nhz
    real_time_scale = 0.9 + (0.1 * random.random())
    real_period = float(message_period) * real_time_scale

    nominal_times = np.arange(start_time_nominal,
                              start_time_nominal + (message_period * num_messages),
                              message_period)

    server_times = np.linspace(start_time_server,
                               start_time_server + (real_period * num_messages),
                               num=num_messages,
                               endpoint=False)

    num_gaps = num_messages // 107
    messages_between_gaps = 106

    for i in range(1, num_gaps):
        gap_i = i * messages_between_gaps
        gap_length = random.randint(1, 10000) * (10 ** 6)
        nominal_times[gap_i:] += gap_length
        server_times[gap_i:] += (float(gap_length) * real_time_scale)

    return nominal_times, server_times.astype(dtype=np.int64)


if __name__ == "__main__":
    [generate_random_sdk_compliant_data(0) for _ in range(10)]
    pass
