from typing import List


def join_sql_and_bools(bool_list: List[str]):
    if not bool_list:
        return ""
    return " WHERE " + " AND ".join(bool_list)
