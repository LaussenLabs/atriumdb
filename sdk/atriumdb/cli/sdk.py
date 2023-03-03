from dotenv import load_dotenv
import os
from atriumdb import AtriumSDK
from atriumdb.sql_handler.sql_constants import SUPPORTED_DB_TYPES

load_dotenv()


def get_sdk_from_env_vars():
    connection_params, dataset_location, metadata_connection_type = get_sdk_params_from_env_vars()

    sdk = AtriumSDK(dataset_location=dataset_location,
                    metadata_connection_type=metadata_connection_type,
                    connection_params=connection_params)
    return sdk


def create_sdk_from_env_vars():
    connection_params, dataset_location, metadata_connection_type = get_sdk_params_from_env_vars()
    sdk = AtriumSDK.create_dataset(dataset_location=dataset_location,
                                   database_type=metadata_connection_type,
                                   connection_params=connection_params,
                                   overwrite='ignore')
    return sdk


def get_sdk_params_from_env_vars():
    dataset_location = os.getenv("ATRIUM_SDK_DATASET_LOCATION")
    metadata_connection_type = os.getenv("ATRIUM_SDK_CONNECTION_TYPE")
    connection_params = None
    if metadata_connection_type in ["mariadb", "mysql"]:
        connection_params = {
            'host': os.getenv("ATRIUM_SDK_METADATA_HOST"),
            'user': os.getenv("ATRIUM_SDK_METADATA_USER"),
            'password': os.getenv("ATRIUM_SDK_METADATA_PASSWORD"),
            'database': os.getenv("ATRIUM_SDK_METADATA_DATABASE"),
            'port': int(os.getenv("ATRIUM_SDK_METADATA_PORT"))
        }
    assert dataset_location is not None, "Must set env var ATRIUM_SDK_DATASET_LOCATION"
    assert metadata_connection_type is not None, "Must set env var ATRIUM_SDK_CONNECTION_TYPE"
    assert metadata_connection_type in SUPPORTED_DB_TYPES, \
        f"ATRIUM_SDK_CONNECTION_TYPE must be one of {SUPPORTED_DB_TYPES}"
    return connection_params, dataset_location, metadata_connection_type
