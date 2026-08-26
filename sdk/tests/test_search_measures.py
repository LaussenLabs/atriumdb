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
"""``search_measures`` must work without being told a frequency.

Every call that passed neither ``freq`` nor ``period`` raised
``TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'``: the branch that
converts a period ran as the ``else`` of the frequency check, so it multiplied ``None``
whenever no frequency was supplied. Searching by tag or unit alone -- the ordinary case,
since a caller who already knew the frequency would not need to search -- was therefore
impossible.

It survived because the only example in the documentation
(``sdk.search_measures(freq=60, freq_units="Hz")``, in the CLI chapter) is the one form
that worked, and the method had no test at all.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \\
        atriumdb-test:latest python -m pytest \\
        /atriumdb/sdk/tests/test_search_measures.py -q
"""
import pytest

from atriumdb import AtriumSDK
from tests.testing_framework import parametrized_backends, prepare_backend

DB_NAME = 'atrium-search-measures'


def _dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk.insert_measure('heart_rate', 1, freq_units='Hz', units='bpm')
    sdk.insert_measure('resp_rate', 1, freq_units='Hz', units='insp/min')
    sdk.insert_measure('ecg_ii', 500, freq_units='Hz', units='mV')
    return sdk


@pytest.mark.parametrize("backend", parametrized_backends())
def test_search_by_tag_alone(backend):
    sdk = _dataset(*prepare_backend(DB_NAME, backend))
    result = sdk.search_measures(tag_match='rate')
    assert {info['tag'] for info in result.values()} == {'heart_rate', 'resp_rate'}
    sdk.close()


@pytest.mark.parametrize("backend", parametrized_backends())
def test_search_by_unit_alone(backend):
    sdk = _dataset(*prepare_backend(DB_NAME + '_unit', backend))
    result = sdk.search_measures(unit='mV')
    assert {info['tag'] for info in result.values()} == {'ecg_ii'}
    sdk.close()


@pytest.mark.parametrize("backend", parametrized_backends())
def test_search_with_no_criteria_returns_everything(backend):
    sdk = _dataset(*prepare_backend(DB_NAME + '_all', backend))
    assert len(sdk.search_measures()) == len(sdk.get_all_measures()) == 3
    sdk.close()


@pytest.mark.parametrize("backend", parametrized_backends())
def test_frequency_and_period_still_filter(backend):
    """The forms that already worked must keep working -- and a frequency filter must
    still exclude, or the fix would have turned every search into `get_all_measures`."""
    sdk = _dataset(*prepare_backend(DB_NAME + '_freq', backend))

    by_freq = sdk.search_measures(freq=500, freq_units='Hz')
    assert {info['tag'] for info in by_freq.values()} == {'ecg_ii'}

    by_period = sdk.search_measures(period=2, time_units='ms')  # 2 ms -> 500 Hz
    assert {info['tag'] for info in by_period.values()} == {'ecg_ii'}

    # Combined criteria still intersect rather than union.
    assert sdk.search_measures(tag_match='rate', freq=500, freq_units='Hz') == {}
    sdk.close()
