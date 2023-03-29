from fastapi import FastAPI

from tests.mock_api.sdk_endpoints import sdk_router

app = FastAPI()
app.include_router(sdk_router, prefix="/v1/sdk")
