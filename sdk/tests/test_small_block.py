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

from atriumdb import AtriumSDK, T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE
import numpy as np
import random

from tests.generate_wfdb import get_records
from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both

DB_NAME = 'small-block'

MAX_RECORDS = 1
SEED = 42


def test_small_block():
    _test_for_both(DB_NAME, _test_small_block)


def _test_small_block(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    one_p_record = next(get_records(dataset_name='mitdb'))
    one_d_record = next(get_records(dataset_name='mitdb', physical=False))

    print()
    print(list(one_p_record.__dict__.keys()))

    for key in one_p_record.__dict__.keys():
        value = one_p_record.__dict__[key]
        print(f"{key}: {value}")

    print(one_d_record.d_signal.T[0])
    print((one_p_record.p_signal.T[0] * one_p_record.adc_gain[0]) + one_p_record.adc_zero[0])

    m2 = (1 / one_p_record.adc_gain[0])
    b2 = (-one_p_record.adc_zero[0] / one_p_record.adc_gain[0])

    print(m2 * one_d_record.d_signal.T[0] + b2)
    print(one_p_record.p_signal.T[0])

    for x, y in zip(get_records(), get_records(physical=False)):
        print(x.p_signal)
        print(y.d_signal)
        print()
