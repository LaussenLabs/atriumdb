# insert_measure_func_query = """
# DROP FUNCTION IF EXISTS insert_measure;
# DELIMITER //
# CREATE FUNCTION insert_measure (freq_nhz BIGINT, measure_tag VARCHAR(64), units VARCHAR(64), measure_name VARCHAR(190))
# RETURNS INT
# BEGIN
#     DECLARE id INT;
#     SELECT id INTO id FROM measures WHERE measure_tag = measure_tag AND freq_nhz = freq_nhz AND units = units;
#     IF id IS NOT NULL THEN
#         RETURN id;
#     ELSE
#         INSERT INTO measures (freq_nhz, measure_tag, units, measure_name) VALUES (freq_nhz, measure_tag, units, measure_name);
#         SELECT LAST_INSERT_ID() INTO id;
#         RETURN id;
#     END IF;
# END; //
# DELIMITER ;
# """

# def get_drop_insert_measure_function(database_name):
#     drop_insert_measure_function = f"DROP FUNCTION IF EXISTS `{database_name}`.insert_measure"
#     return drop_insert_measure_function
#
#
# def get_insert_measure_func_query(database_name):
#     insert_measure_func_query = f"""
#             CREATE FUNCTION `{database_name}`.insert_measure (freq_nhz BIGINT, measure_tag VARCHAR(64), units VARCHAR(64), measure_name VARCHAR(190))
#             RETURNS INT
#             BEGIN
#                 DECLARE id INT;
#                 SELECT id INTO id FROM measures WHERE measure_tag = measure_tag AND freq_nhz = freq_nhz AND units = units;
#                 IF id IS NOT NULL THEN
#                     RETURN id;
#                 ELSE
#                     INSERT INTO measures (freq_nhz, measure_tag, units, measure_name) VALUES (freq_nhz, measure_tag, units, measure_name);
#                     SELECT LAST_INSERT_ID() INTO id;
#                     RETURN id;
#                 END IF;
#             END;
#             """
#     return insert_measure_func_query

drop_insert_measure_function = "DROP FUNCTION IF EXISTS insert_measure"


insert_measure_func_query = """
            CREATE FUNCTION insert_measure (freq_nhz BIGINT, measure_tag VARCHAR(64), units VARCHAR(64), measure_name VARCHAR(190))
            RETURNS INT
            BEGIN
                DECLARE id INT;
                SELECT id INTO id FROM measures WHERE measure_tag = measure_tag AND freq_nhz = freq_nhz AND units = units;
                IF id IS NOT NULL THEN
                    RETURN id;
                ELSE
                    INSERT INTO measures (freq_nhz, measure_tag, units, measure_name) VALUES (freq_nhz, measure_tag, units, measure_name);
                    SELECT LAST_INSERT_ID() INTO id;
                    RETURN id;
                END IF;
            END;
            """

select_measure_from_id_query = "SELECT * FROM measures WHERE id = ?"
select_measure_from_triplet_query = "SELECT * FROM measures WHERE measure_tag = ? AND freq_nano = ? AND units = ?"