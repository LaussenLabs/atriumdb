import mariadb

from atriumdb.sql_handler.maria.maria_functions import maria_select_measure_from_triplet_query, \
    maria_select_measure_from_id_query, \
    maria_insert_ignore_measure_query, maria_insert_ignore_device_query, maria_select_device_from_tag_query, maria_select_device_from_id_query
from atriumdb.sql_handler.maria.maria_tables import maria_measures_table_create_query, maria_devices_table_create_query, \
    maria_file_index_create_query, maria_block_index_create_query, maria_interval_index_create_query, maria_encounter_create_query, \
    maria_settings_create_query
from atriumdb.sql_handler.sql_constants import DEFAULT_UNITS
from atriumdb.sql_handler.sql_handler import SQLHandler


class MariaDBHandler(SQLHandler):
    def __init__(self, host: str, user: str, password: str, dbname: str, port: int = 3306):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.dbname = dbname

    def maria_connect(self, db_name=None):
        conn = mariadb.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=db_name)

        return conn

    def create_schema(self):
        conn = self.maria_connect()
        cursor = conn.cursor()

        # Create Schema
        cursor.execute("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))
        cursor.close()
        conn.change_user(self.user, self.password, self.dbname)
        cursor = conn.cursor()

        # Create Tables
        cursor.execute(maria_measures_table_create_query)
        cursor.execute(maria_devices_table_create_query)
        cursor.execute(maria_file_index_create_query)
        cursor.execute(maria_block_index_create_query)
        cursor.execute(maria_interval_index_create_query)
        cursor.execute(maria_encounter_create_query)
        cursor.execute(maria_settings_create_query)

        conn.commit()
        cursor.close()
        conn.close()

    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None):
        units = DEFAULT_UNITS if units is None else units

        conn = self.maria_connect(db_name=self.dbname)
        cursor = conn.cursor()

        try:
            cursor.execute(maria_insert_ignore_measure_query, (measure_tag, freq_nhz, units, measure_name))
            conn.commit()
            cursor.execute(maria_select_measure_from_triplet_query, (measure_tag, freq_nhz, units))
            measure_id = cursor.fetchone()[0]
        except mariadb.Error as e:
            raise e
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return measure_id

    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        units = DEFAULT_UNITS if units is None else units
        conn = self.maria_connect(db_name=self.dbname)
        cursor = conn.cursor()

        try:
            if measure_id is not None:
                cursor.execute(maria_select_measure_from_id_query, (measure_id,))
            else:
                cursor.execute(maria_select_measure_from_triplet_query,
                               (measure_tag, freq_nhz, units))
            row = cursor.fetchone()
        except mariadb.Error as e:
            raise e
        finally:
            cursor.close()
            conn.close()
        return row

    def insert_device(self, device_tag: str, device_name: str = None):
        conn = self.maria_connect(db_name=self.dbname)
        cursor = conn.cursor()

        try:
            cursor.execute(maria_insert_ignore_device_query, (device_tag, device_name))
            conn.commit()
            cursor.execute(maria_select_device_from_tag_query, (device_tag,))
            device_id = cursor.fetchone()[0]
        except mariadb.Error as e:
            raise e
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return device_id

    def select_device(self, device_id: int = None, device_tag: str = None):
        conn = self.maria_connect(db_name=self.dbname)
        cursor = conn.cursor()

        try:
            if device_id is not None:
                cursor.execute(maria_select_device_from_id_query, (device_id,))
            else:
                cursor.execute(maria_select_device_from_tag_query, (device_tag,))
            row = cursor.fetchone()
        except mariadb.Error as e:
            raise e
        finally:
            cursor.close()
            conn.close()
        return row


