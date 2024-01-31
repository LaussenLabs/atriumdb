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

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'sql_label_set_parents'


def test_sql_label_set_parents():
    _test_for_both(DB_NAME, _test_sql_label_set_parents)


def _test_sql_label_set_parents(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # SQLHandler object
    sql_handler = sdk.sql_handler

    # Creating a tree structure
    root_id = sql_handler.insert_label_set("Root")
    child1_id = sql_handler.insert_label_set("Child1", parent_id=root_id)
    child2_id = sql_handler.insert_label_set("Child2", parent_id=root_id)
    grandchild1_id = sql_handler.insert_label_set("GrandChild1", parent_id=child1_id)
    grandchild2_id = sql_handler.insert_label_set("GrandChild2", parent_id=child1_id)
    isolated_node_id = sql_handler.insert_label_set("IsolatedNode")

    # Test select_parent
    assert sql_handler.select_label_name_parent(child1_id) == (root_id, "Root"), "select_parent failed for Child1"
    assert sql_handler.select_label_name_parent(isolated_node_id) is None, "select_parent failed for IsolatedNode"

    # Test select_all_ancestors
    assert set(sql_handler.select_all_ancestors(grandchild1_id)) == {(root_id, "Root"), (child1_id, "Child1")}, "select_all_ancestors failed for GrandChild1"
    assert sql_handler.select_all_ancestors(isolated_node_id) == [], "select_all_ancestors failed for IsolatedNode"

    # Test select_children
    assert set(sql_handler.select_label_name_children(root_id)) == {(child1_id, "Child1"), (child2_id, "Child2")}, "select_children failed for Root"
    assert sql_handler.select_label_name_children(grandchild1_id) == [], "select_children failed for GrandChild1"

    # Test select_all_descendants
    assert set(sql_handler.select_all_label_name_descendents(root_id)) == {(child1_id, "Child1", root_id), (child2_id, "Child2", root_id), (grandchild1_id, "GrandChild1", child1_id), (grandchild2_id, "GrandChild2", child1_id)}, "select_all_descendants failed for Root"
    assert sql_handler.select_all_label_name_descendents(
        isolated_node_id) == [], "select_all_descendants failed for IsolatedNode"
