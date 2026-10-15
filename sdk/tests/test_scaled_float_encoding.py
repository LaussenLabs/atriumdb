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

from atriumdb import AtriumSDK
from atriumdb.helpers.block_constants import VALUE_TYPES
from tests.testing_framework import _test_for_both

DB_NAME = 'scaled_float'


def test_scaled_float_encoding():
    _test_for_both(DB_NAME, _test_scaled_float_encoding)


def _test_scaled_float_encoding(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk.block.block_size = 4096

    rng = np.random.default_rng(42)
    adc = (rng.integers(-2048, 2048, 20_000) + 1024) / 200.0
    adc_with_nan = adc.copy()
    adc_with_nan[5000] = np.nan
    scaled, double = VALUE_TYPES['SCALED_INT64'], VALUE_TYPES['DOUBLE']

    cases = {
        # ADC samples in physical units, as wfdb computes them.
        "adc": (adc, [scaled] * 4),
        # Decimal readings, such as a temperature written as text.
        "decimal": (np.round(rng.normal(37, 1, 20_000), 2), [scaled] * 4),
        # No scale exists.
        "noise": (rng.normal(0, 1, 20_000), [double] * 4),
        # Only the block with the NaN stays doubles.
        "adc_with_nan": (adc_with_nan, [scaled, double, scaled, scaled]),
    }
    for tag, (values, encoded_types) in cases.items():
        measure_id = sdk.insert_measure(measure_tag=tag, freq=500, freq_units="Hz")
        device_id = sdk.insert_device(device_tag=tag)
        sdk.write_segment(measure_id, device_id, values, 0, freq=500, freq_units="Hz", time_units="s")

        headers, _, r_values = sdk.get_data(measure_id, 0, 41 * 10 ** 9, device_id=device_id)

        assert [h.v_encoded_type for h in headers] == encoded_types
        assert np.array_equal(r_values.view(np.int64), values.view(np.int64))
