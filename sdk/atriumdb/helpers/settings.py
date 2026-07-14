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

OVERWRITE_SETTING_NAME = 'overwrite'
PROTECTED_MODE_SETTING_NAME = 'protected_mode'

# The 'overwrite' setting is the dataset's merge conflict policy: how block
# merging resolves duplicate timestamps between a new write and existing data.
# 'overwrite' and the legacy default 'ignore' keep the new write's values,
# 'protect' keeps the existing values, and 'error' refuses to merge writes that
# conflict with existing data. See AtriumSDK._merge_conflict_policy.
ALLOWABLE_OVERWRITE_SETTINGS = ['error', 'ignore', 'overwrite', 'protect']
ALLOWABLE_PROTECTED_MODE_SETTINGS = ['True', 'False']
