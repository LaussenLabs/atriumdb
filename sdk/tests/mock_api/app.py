from fastapi import FastAPI

from tests.mock_api.devices_endpoints import devices_router
from tests.mock_api.measures_endpoints import measures_router
from tests.mock_api.sdk_endpoints import sdk_router

app = FastAPI()
app.include_router(sdk_router, prefix="/sdk")
app.include_router(measures_router, prefix="/measures")
app.include_router(devices_router, prefix="/devices")
