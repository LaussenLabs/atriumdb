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
"""Legacy ``freq_nhz = 0`` measures: an aperiodic signal in the old vocabulary.

Before ``signal_kind`` existed, a writer said "this signal is aperiodic" by giving the
measure a frequency of zero -- an intermittent measurement such as a non-invasive blood
pressure cuff, which has no sampling rate to state. The *data* those writers produced is
already what this SDK calls an aperiodic sample measure: an explicit timestamp array,
with the period left unset at write time so it was detected from the timestamps and
recorded in each block header. Only the ``measure`` row speaks the old language, so
bringing it forward is a rename, not a re-encode.

Two things are covered here.

**The row must not be fatal.** ``AtriumSDK.__init__`` caches every measure and computed
``10 ** 18 // freq_nhz``, so one such row raised ``ZeroDivisionError`` from inside the
constructor: the dataset could not be opened at all, could not be repaired through the
library, and ``auto_upgrade=True`` could not help because the crash preceded it. A
measure with no usable frequency now reports no period and warns.

**``auto_upgrade=True`` converts it.** The row gains ``signal_kind='sample'`` and the
nominal period its own blocks demonstrate, derived from ``block_index`` alone -- the
repair runs mid-construction, before the file API and codec are usable.

The fixtures below use raw SQL for the zero-frequency row on purpose: ``insert_measure``
rejects ``freq <= 0``, so this shape can no longer be created through the API, and raw
SQL is the only way to reproduce what a pre-guard writer left behind.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \\
        atriumdb-test:latest python -m pytest \\
        /atriumdb/sdk/tests/test_legacy_zero_freq_measure.py -q
"""
import numpy as np
import pytest

from atriumdb import AtriumSDK
from tests.testing_framework import parametrized_backends, prepare_backend

DB_NAME = 'atrium-legacy-zero-freq'

SEC = 10 ** 9
# A cuff reading every five minutes: sparse, irregular in principle, and nothing a
# frequency describes well -- which is why the old marker existed.
NIBP_PERIOD_NS = 300 * SEC
NIBP_SAMPLES = 20
BASE = 1_700_000_000 * SEC


def _legacy_dataset(db_type, dataset_location, connection_params):
    """A dataset holding one aperiodic measure expressed the old way.

    Written normally so the blocks are real, then the measure row is rewound to the
    legacy shape: frequency zero, no period, no declared kind.
    """
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    measure_id = sdk.insert_measure('nibp_sys', 1 / 300, freq_units='Hz', units='mmHg')
    device_id = sdk.insert_device('dev_cuff')

    times = BASE + np.arange(NIBP_SAMPLES, dtype=np.int64) * NIBP_PERIOD_NS
    values = np.linspace(110.0, 130.0, NIBP_SAMPLES, dtype=np.float64)
    sdk.write_time_value_pairs(measure_id, device_id, times, values)

    with sdk.sql_handler.connection(begin=True) as (conn, cursor):
        cursor.execute(
            "UPDATE measure SET freq_nhz = 0, period_ns = NULL, signal_kind = NULL WHERE id = ?",
            (int(measure_id),))
    sdk.close()
    return measure_id, device_id, times, values


@pytest.mark.parametrize("backend", parametrized_backends())
def test_legacy_zero_freq_measure_does_not_make_the_dataset_unopenable(backend):
    db_type, dataset_location, connection_params = prepare_backend(DB_NAME, backend)
    measure_id, device_id, times, values = _legacy_dataset(db_type, dataset_location, connection_params)

    # The default auto_upgrade=False. Opening must work: a dataset you cannot open is
    # one you cannot repair.
    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                    connection_params=connection_params)
    measures = sdk.get_all_measures()
    assert measure_id in measures, "the legacy measure must still be discoverable"
    assert measures[measure_id]['period_ns'] is None, \
        "no period can honestly be reported for a zero-frequency row"
    sdk.close()


@pytest.mark.parametrize("backend", parametrized_backends())
def test_auto_upgrade_converts_legacy_zero_freq_to_an_aperiodic_sample_measure(backend):
    db_type, dataset_location, connection_params = prepare_backend(DB_NAME + '_conv', backend)
    measure_id, device_id, times, values = _legacy_dataset(db_type, dataset_location, connection_params)

    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                    connection_params=connection_params, auto_upgrade=True)

    info = sdk.get_measure_info(measure_id)
    assert info['signal_kind'] == 'sample', "an aperiodic measure, now said outright"
    assert info['period_ns'] == NIBP_PERIOD_NS, (
        f"the nominal period should come from the measure's own blocks; "
        f"got {info['period_ns']}, expected {NIBP_PERIOD_NS}")
    assert info['freq_nhz'] == (10 ** 18) // NIBP_PERIOD_NS

    # The conversion is metadata only -- every stored sample reads back unchanged.
    _, read_times, read_values = sdk.get_data(
        measure_id, int(times[0]) - SEC, int(times[-1]) + SEC, device_id=device_id)
    order = np.argsort(read_times)
    np.testing.assert_array_equal(read_times[order], times)
    np.testing.assert_allclose(read_values[order], values)
    sdk.close()

    # Idempotent: a second run finds nothing to convert and changes nothing.
    again = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                      connection_params=connection_params, auto_upgrade=True)
    assert again.sql_handler.select_zero_freq_measures() == []
    assert again.get_measure_info(measure_id)['period_ns'] == NIBP_PERIOD_NS
    again.close()


@pytest.mark.parametrize("backend", parametrized_backends())
def test_conversion_leaves_a_colliding_measure_alone(backend):
    """`UNIQUE (tag, freq_nhz, unit)` means the converted frequency can already be
    taken. Merging two measures is a data decision, not a migration's to make, so the
    legacy row is left as it is rather than silently folded into the other."""
    db_type, dataset_location, connection_params = prepare_backend(DB_NAME + '_collide', backend)
    measure_id, device_id, _times, _values = _legacy_dataset(db_type, dataset_location, connection_params)

    # Occupy the frequency the conversion would want, under the same tag and unit.
    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                    connection_params=connection_params)
    with sdk.sql_handler.connection(begin=True) as (conn, cursor):
        cursor.execute(
            "INSERT INTO measure (tag, freq_nhz, unit) VALUES ('nibp_sys', ?, 'mmHg')",
            ((10 ** 18) // NIBP_PERIOD_NS,))
    sdk.close()

    upgraded = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                         connection_params=connection_params, auto_upgrade=True)
    assert [row[0] for row in upgraded.sql_handler.select_zero_freq_measures()] == [measure_id], \
        "the colliding legacy row must be left for a human to resolve"
    upgraded.close()
