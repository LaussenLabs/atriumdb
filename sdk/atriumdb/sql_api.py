from atriumdb.helpers import sql_helpers
from atriumdb.helpers.sql_tables import generate_metadata, MEASURE_TABLE_NAME, SETTINGS_TABLE_NAME, DEVICE_TABLE_NAME, \
    BLOCK_TABLE_NAME, FILE_TABLE_NAME
from sqlalchemy import *
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
# Uncomment this code when sqlalchemy-utils becomes compatible with sqlalchemy 1.4 or greater.
# from sqlalchemy_utils import database_exists, create_database

supported_db_types = ["mysql", "sqlite"]
# mariadb driver: mysqlclient

measure_optional_vars = ['id', 'measure_tag', 'measure_name', 'units']
device_optional_vars = ['id', 'device_tag', 'device_name']

DEFAULT_DB_URI = 'sqlite://'  # In Memory Only


class AtriumSql:

    def __init__(self, db_uri: str = None):
        db_uri = DEFAULT_DB_URI if db_uri is None else db_uri
        db_type = db_uri.split(':')[0].split('+')[0]
        self.db_type = db_type

        if db_type not in supported_db_types:
            raise ValueError("db_type {} not in {}.".format(db_type, supported_db_types))

        engine_kwargs = {}

        if db_type == "mysql":
            engine_kwargs['isolation_level'] = "READ COMMITTED"
            # engine_kwargs['pool_recycle'], engine_kwargs['pool_size'] = 60, 1
            # engine_kwargs['pool_pre_ping'] = True
            engine_kwargs['poolclass'] = NullPool

        elif db_type == "sqlite":
            pass

        else:
            raise ValueError("db_type {} not in {}.".format(db_type, supported_db_types))

        self.engine = create_engine(
            db_uri,
            future=True,
            # echo=True,
            **engine_kwargs)

        # Uncomment this code when sqlalchemy-utils becomes compatible with sqlalchemy 1.4 or greater.
        # if not database_exists(self.engine.url):
        #     create_database(self.engine.url)

        self.metadata = generate_metadata()
        self.metadata.create_all(self.engine)

        self.measure_id_dict, self.measure_tag_dict = None, None
        self.device_id_dict, self.device_tag_dict = None, None
        self.populate_measure_device_cache()

    def insert_setting(self, connection, setting_name, setting_value):
        settings_table = self.metadata.tables[SETTINGS_TABLE_NAME]

        stmt = insert(settings_table).values(
            setting_name=setting_name, setting_value=setting_value)
        connection.execute(stmt)

        return

    def update_setting(self, connection, setting_name, setting_value):
        settings_table = self.metadata.tables[SETTINGS_TABLE_NAME]

        with self.lock_tables(connection, [SETTINGS_TABLE_NAME]):
            stmt = update(settings_table).where(
                settings_table.c.setting_name == setting_name).values(
                setting_value=setting_value)

            connection.execute(stmt)

        return

    def select_all_settings(self, connection):
        settings_table = self.metadata.tables[SETTINGS_TABLE_NAME]

        stmt = select(settings_table)

        return connection.execute(stmt).all()

    def unlock_tables(self, connection):
        if self.db_type == 'mysql':
            connection.execute(text("UNLOCK TABLES;"))
        if self.db_type == 'sqlite':
            connection.execute(text("COMMIT;"))

    @contextmanager
    def lock_tables(self, connection, table_name_list):
        try:
            if self.db_type == 'mysql':
                assert len(table_name_list) > 0
                lock_stmt = "LOCK TABLES "
                lock_stmt += ', '.join([f"`{table_name}` WRITE" for table_name in table_name_list])
                lock_stmt += ";"
                connection.execute(text(lock_stmt))
            if self.db_type == 'sqlite':
                connection.execute(text("BEGIN EXCLUSIVE;"))

            with self.transaction(connection):
                yield connection
            self.unlock_tables(connection)
        finally:
            pass

    @contextmanager
    def transaction(self, connection):
        if not connection.in_transaction():
            with connection.begin():
                yield connection
        else:
            yield connection

    def populate_measure_device_cache(self):
        dict_result = self.pull_measure_device_dicts()
        self.measure_id_dict, self.measure_tag_dict = dict_result[:2]
        self.device_id_dict, self.device_tag_dict = dict_result[2:]

    def get_measure_id(self, measure_tag, freq, units):
        # check if the measure_tag, freq pair is in the cached measure_tag_dict
        if (measure_tag, freq, units) in self.measure_tag_dict:
            return self.measure_tag_dict[(measure_tag, freq, units)]

        # if it's not in the dictionary repopulate the dict as the dictionary is only populated when the sdk object is
        # initially created so there may be more measure tags in the sql database
        self.populate_measure_device_cache()

        # check again and if its not there then return none
        if (measure_tag, freq, units) in self.measure_tag_dict:
            return self.measure_tag_dict[(measure_tag, freq, units)]
        else:
            return None

    def get_device_id(self, device_tag):
        # check if the measure_tag, freq pair is in the cached measure_tag_dict
        if device_tag in self.device_tag_dict:
            return self.device_tag_dict[device_tag]

        # if it's not in the dictionary repopulate the dict as the dictionary is only populated when the sdk object is
        # initially created so there may be more device tags in the sql database
        self.populate_measure_device_cache()

        # check again and if its not there then return none
        if device_tag in self.device_tag_dict:
            return self.device_tag_dict[device_tag]
        else:
            return None

    def get_measure_interval_dict(self, device_id: int, start_time_nano: int, end_time_nano: int):
        """
        Creates a dictionary with keys beings the measure_ids available for the given device.
        And dictionary values being the time intervals for which data is available.
        """
        with self.engine.connect() as conn:
            query = select([self.metadata.tables['interval_index'].c.measure_id,
                            self.metadata.tables['interval_index'].c.start_time_n,
                            self.metadata.tables['interval_index'].c.end_time_n]).where(
                and_(self.metadata.tables['interval_index'].c.device_id == device_id,
                     self.metadata.tables['interval_index'].c.start_time_n < end_time_nano,
                     self.metadata.tables['interval_index'].c.end_time_n > start_time_nano))
            result = conn.execute(query)
            result_dict = {}
            for row in result:
                if row[0] in result_dict:
                    result_dict[row[0]].append((row[1], row[2]))
                else:
                    result_dict[row[0]] = [(row[1], row[2])]
            return result_dict

    def pull_measure_device_dicts(self):
        measure_id_dict, measure_tag_dict, device_id_dict, device_tag_dict = {}, {}, {}, {}
        with self.connect() as conn:
            measure_result = self.select_measure_id(conn)
            for row in measure_result:
                measure_id_dict[row.id] = {'measure_tag': row.measure_tag, 'measure_name': row.measure_name,
                                           'freq_nhz': row.freq_nhz, 'units': row.units}
                measure_tag_dict[(row.measure_tag, row.freq_nhz, row.units)] = row.id

            device_result = self.select_device_id(conn)
            for row in device_result:
                device_id_dict[row.id] = {'device_tag': row.device_tag, 'device_name': row.device_name}
                device_tag_dict[row.device_tag] = row.id

        return measure_id_dict, measure_tag_dict, device_id_dict, device_tag_dict

    def connect(self):
        return self.engine.connect()

    def begin(self):
        return self.engine.begin()

    def insert_block_index(self, connection, measure_id, device_id, file_id, start_byte,
                           num_bytes, start_time_n, end_time_n, num_values):

        block_index_table = self.metadata.tables[BLOCK_TABLE_NAME]
        stmt = insert(block_index_table).values(
            measure_id=measure_id, device_id=device_id, file_id=file_id, start_byte=start_byte,
            num_bytes=num_bytes, start_time_n=start_time_n, end_time_n=end_time_n, num_values=num_values)

        return connection.execute(stmt).inserted_primary_key[0]

    def insert_interval_index(self, connection, measure_id, device_id, start_time_n, end_time_n):
        interval_index_table = self.metadata.tables['interval_index']
        stmt = insert(interval_index_table).values(
            measure_id=measure_id, device_id=device_id, start_time_n=start_time_n, end_time_n=end_time_n)

        return connection.execute(stmt).inserted_primary_key[0]

    def insert_file_index(self, connection, measure_id, device_id, path):
        file_table = self.metadata.tables['file_index']
        stmt = insert(file_table).values(
            measure_id=measure_id, device_id=device_id, path=path)

        return connection.execute(stmt).inserted_primary_key[0]

    def delete_block_id(self, connection, block_id):
        block_table = self.metadata.tables[BLOCK_TABLE_NAME]
        stmt = delete(block_table).where(
            block_table.c.id == block_id
        )
        connection.execute(stmt)

    def delete_file_id(self, connection, file_id):
        file_table = self.metadata.tables[FILE_TABLE_NAME]
        stmt = delete(file_table).where(
            file_table.c.id == file_id
        )
        connection.execute(stmt)

    def insert_measure_id(self, connection, freq_nhz, **kwargs):
        if not all(key in measure_optional_vars for key in kwargs.keys()):
            raise ValueError("Allowable keyword arguments are: {}".format(measure_optional_vars))

        measure_table = self.metadata.tables[MEASURE_TABLE_NAME]
        with self.lock_tables(connection, [MEASURE_TABLE_NAME]):

            select_stmt = select(measure_table.c.id).where(
                (measure_table.c.measure_tag == kwargs['measure_tag']) & (measure_table.c.freq_nhz == freq_nhz))

            measure_id = connection.execute(select_stmt).one_or_none()

            if measure_id is not None:
                return measure_id

            stmt = insert(measure_table).values(
                freq_nhz=freq_nhz, **kwargs)

            measure_id = connection.execute(stmt).inserted_primary_key[0]

        return measure_id

    def select_measure_id(self, connection):
        measure_table = self.metadata.tables['measures']

        stmt = select(measure_table)
        return connection.execute(stmt)

    def insert_device_id(self, connection, **kwargs):
        if not all(key in device_optional_vars for key in kwargs.keys()):
            raise ValueError("Allowable keyword arguments are: {}".format(device_optional_vars))

        device_tables = self.metadata.tables[DEVICE_TABLE_NAME]
        with self.lock_tables(connection, [DEVICE_TABLE_NAME]):

            select_stmt = select(device_tables.c.id).where(
                device_tables.c.device_tag == kwargs['device_tag'])

            device_id = connection.execute(select_stmt).one_or_none()

            if device_id is not None:
                return device_id

            stmt = insert(device_tables).values(**kwargs)

            device_id = connection.execute(stmt).inserted_primary_key[0]

        return device_id

    def select_device_id(self, connection):
        device_tables = self.metadata.tables['devices']

        stmt = select(device_tables)
        return connection.execute(stmt)

    def select_blocks_from_file(self, connection, file_id):
        block_index_table = self.metadata.tables[BLOCK_TABLE_NAME]
        stmt = select(block_index_table).where(
            block_index_table.c.file_id == file_id).order_by(
            block_index_table.c.start_byte)

        return connection.execute(stmt)

    def select_block_ids(self, connection, measure_id, start_time_n=None, end_time_n=None, device_id=None,
                         patient_id=None):
        identifiers = [device_id, patient_id]
        if all([identifier is None for identifier in identifiers]):
            raise ValueError("One of device_id, patient_id must be specified")

        if device_id is not None:
            block_index_table = self.metadata.tables[BLOCK_TABLE_NAME]

            # stmt = select(block_index_table).where(
            #     block_index_table.c.measure_id == measure_id).where(
            #     block_index_table.c.device_id == device_id).where(
            #     block_index_table.c.start_time_n <= end_time_n).where(
            #     block_index_table.c.end_time_n >= start_time_n).order_by(
            #     block_index_table.c.file_id, block_index_table.c.start_byte)

            stmt = select(block_index_table).where(
                block_index_table.c.measure_id == measure_id).where(
                block_index_table.c.device_id == device_id)

            if end_time_n is not None:
                stmt = stmt.where(block_index_table.c.start_time_n <= end_time_n)

            if start_time_n is not None:
                stmt = stmt.where(block_index_table.c.end_time_n >= start_time_n)

            stmt = stmt.order_by(
                block_index_table.c.file_id, block_index_table.c.start_byte)

        elif patient_id is not None:
            block_index_table = self.metadata.tables['block_index']
            encounter_table = self.metadata.tables['encounter']

            # encounter_query = select([encounter_table]).where(
            #     encounter_table.c.patient_id == patient_id).where(
            #     encounter_table.c.start_time_n <= end_time_n).where(
            #     encounter_table.c.end_time_n >= start_time_n).order_by(
            #     encounter_table.c.patient_id, encounter_table.c.start_time_n)

            encounter_query = select([encounter_table]).where(
                encounter_table.c.patient_id == patient_id)

            if end_time_n is not None:
                encounter_query = encounter_query.where(encounter_table.c.start_time_n <= end_time_n)

            if start_time_n is not None:
                encounter_query = encounter_query.where(encounter_table.c.end_time_n >= start_time_n)

            encounter_query = encounter_query.order_by(
                encounter_table.c.patient_id, encounter_table.c.start_time_n)

            # block_query = select([block_index_table]).where(
            #     block_index_table.c.measure_id == measure_id).where(
            #     block_index_table.c.start_time_n <= end_time_n).where(
            #     block_index_table.c.end_time_n >= start_time_n).order_by(
            #     block_index_table.c.file_id, block_index_table.c.start_byte)

            block_query = select([block_index_table]).where(
                block_index_table.c.measure_id == measure_id)

            if end_time_n is not None:
                block_query = block_query.where(block_index_table.c.start_time_n <= end_time_n)

            if start_time_n is not None:
                block_query = block_query.where(block_index_table.c.end_time_n >= start_time_n)

            block_query = block_query.order_by(
                block_index_table.c.file_id, block_index_table.c.start_byte)

            encounter_subquery = encounter_query.subquery()
            block_subquery = block_query.subquery()

            j = join(encounter_subquery, block_subquery, and_(
                encounter_subquery.c.device_id == block_subquery.c.device_id,
                encounter_subquery.c.start_time_n <= block_subquery.c.end_time_n,
                encounter_subquery.c.end_time_n >= block_subquery.c.start_time_n))

            stmt = select([block_subquery]).select_from(j)

        else:
            # Should be unreachable
            raise ValueError("One of {} must be specified".format(identifiers))

        return connection.execute(stmt)

    def select_exists_signal_chunk(self, connection, measure_id, device_id, start_time_nano):
        interval_index = self.metadata.tables['interval_index']

        stmt = exists(interval_index.c.id).where(
            interval_index.c.measure_id == measure_id).where(
            interval_index.c.device_id == device_id).where(
            interval_index.c.start_time_n == start_time_nano).select()

        return connection.execute(stmt)

    def select_block_info_by_id(self, connection, block_id):
        block_index_table = self.metadata.tables['block_index']

        stmt = select(block_index_table).where(
            block_index_table.c.id == block_id)

        return connection.execute(stmt)

    def select_interval_index(self, connection, measure_id, device_id=None, patient_id=None, start=None, end=None):
        interval_index = self.metadata.tables['interval_index']
        encounter_table = self.metadata.tables['encounter']

        if device_id is not None:
            stmt = select(interval_index).where(
                interval_index.c.measure_id == measure_id).where(
                interval_index.c.device_id == device_id)
        elif patient_id is not None:
            encounter_query = select([encounter_table]).where(
                encounter_table.c.patient_id == patient_id)

            interval_query = select(interval_index).where(
                interval_index.c.measure_id == measure_id)

            encounter_subquery = encounter_query.subquery()
            interval_subquery = interval_query.subquery()

            j = join(encounter_subquery, interval_subquery,
                     and_(
                         encounter_subquery.c.device_id == interval_subquery.c.device_id,
                         encounter_subquery.c.start_time_n <= interval_subquery.c.end_time_n,
                         encounter_subquery.c.end_time_n >= interval_subquery.c.start_time_n))

            stmt = select([interval_subquery]).select_from(j)
            interval_index = interval_subquery
        else:
            raise ValueError("One of device_id, patient_id must be specified.")

        if start is not None:
            stmt = stmt.where(interval_index.c.end_time_n >= start)

        if end is not None:
            stmt = stmt.where(interval_index.c.start_time_n <= end)

        stmt = stmt.order_by(interval_index.c.start_time_n)

        return connection.execute(stmt)

    def select_patient_ids(self, connection, start=None, end=None):
        encounter_table = self.metadata.tables['encounter']
        stmt = select(distinct(encounter_table.c.patient_id))

        if start is not None:
            stmt = stmt.where(encounter_table.c.end_time_n >= start)

        if end is not None:
            stmt = stmt.where(encounter_table.c.start_time_n <= end)

        return connection.execute(stmt)

    def select_all_signal_intervals(self, connection, device_id, start=None, end=None):
        interval_index = self.metadata.tables['interval_index']
        stmt = select(interval_index).where(
            interval_index.c.device_id == device_id)

        if start is not None:
            stmt = stmt.where(interval_index.c.end_time_n >= start)

        if end is not None:
            stmt = stmt.where(interval_index.c.start_time_n <= end)

        stmt = stmt.order_by(interval_index.c.start_time_n)

        return connection.execute(stmt)

    def select_file_index(self, connection, measure_id, device_id):
        file_index = self.metadata.tables['file_index']
        stmt = select(file_index).where(
            file_index.c.measure_id == measure_id).where(
            file_index.c.device_id == device_id)

        return connection.execute(stmt)

    def select_filename_dict(self, connection, file_id_list):
        file_index = self.metadata.tables['file_index']
        stmt = select(file_index.c.id, file_index.c.path).where(file_index.c.id.in_(file_id_list))

        result = connection.execute(stmt)
        result_dict = {}
        for row in result:
            result_dict[row.id] = row.path

        return result_dict

    def insert_encounter(self, connection, patient_id, device_id, start_time_n, end_time_n):
        encounter_table = self.metadata.tables['encounter']
        stmt = insert(encounter_table).values(patient_id=patient_id,
                                              device_id=device_id,
                                              start_time_n=start_time_n,
                                              end_time_n=end_time_n)
        return connection.execute(stmt)

    def insert_new_session(self, connection, session_hash):
        session_table = self.metadata.tables['session']
        stmt = insert(session_table).values(hash=session_hash)

        return connection.execute(stmt).inserted_primary_key[0]

    def select_session(self, connection, session_hash):
        session_table = self.metadata.tables['session']
        stmt = select(session_table).where(session_table.c.hash == session_hash)

        return connection.execute(stmt)

    def insert_ingest_packet(self, connection, session_id, packet_hash):
        ingest_table = self.metadata.tables['ingest']
        stmt = insert(ingest_table).values(session_id=session_id, hash=packet_hash)

        return connection.execute(stmt).inserted_primary_key[0]

    def select_session_packets(self, connection, session_id):
        ingest_table = self.metadata.tables['ingest']
        stmt = select(ingest_table.c.hash).where(ingest_table.c.session_id == session_id)

        return connection.execute(stmt)

    def select_available_measures(self, connection, device_id=None, patient_id=None, start=None, end=None):
        interval_index = self.metadata.tables['interval_index']
        encounter_table = self.metadata.tables['encounter']

        if device_id is not None:
            stmt = select(distinct(interval_index.c.measure_id)).where(
                interval_index.c.device_id == device_id)

        elif patient_id is not None:
            encounter_query = select([encounter_table]).where(
                encounter_table.c.patient_id == patient_id)

            interval_query = select(interval_index)

            if end is not None:
                interval_query = interval_query.where(interval_index.c.start_time_n <= end)
                encounter_query = encounter_query.where(encounter_table.c.start_time_n <= end)

            if start is not None:
                interval_query = interval_query.where(interval_index.c.end_time_n >= start)
                encounter_query = encounter_query.where(encounter_table.c.end_time_n >= start)

            encounter_subquery = encounter_query.subquery()
            interval_subquery = interval_query.subquery()

            j = join(encounter_subquery, interval_subquery,
                     and_(
                         encounter_subquery.c.device_id == interval_subquery.c.device_id,
                         encounter_subquery.c.start_time_n <= interval_subquery.c.end_time_n,
                         encounter_subquery.c.end_time_n >= interval_subquery.c.start_time_n))

            stmt = select([distinct(interval_subquery.c.measure_id)]).select_from(j)
            interval_index = interval_subquery

        else:
            raise ValueError("Must specify device or patient id.")

        if end is not None:
            stmt = stmt.where(interval_index.c.start_time_n <= end)

        if start is not None:
            stmt = stmt.where(interval_index.c.end_time_n >= start)

        return connection.execute(stmt)

    def select_available_devices(self, connection, measure_id, start=None, end=None):
        interval_index = self.metadata.tables['interval_index']

        stmt = select(distinct(interval_index.c.device_id)).where(
            interval_index.c.measure_id == measure_id)

        if start is not None:
            stmt = stmt.where(interval_index.c.end_time_n >= start)

        if end is not None:
            stmt = stmt.where(interval_index.c.start_time_n <= end)

        return connection.execute(stmt)
