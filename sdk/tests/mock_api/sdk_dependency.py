from fastapi import Request

from atriumdb import AtriumSDK


def get_sdk_instance() -> AtriumSDK:
    return AtriumSDK()
