"""FastAPI entrypoint for the AI solar planning backend."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from solar_backend.api.router import api_router
from solar_backend.core.config import settings
from solar_backend.core.logging import configure_logging

configure_logging()

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.mount(settings.static_url_prefix, StaticFiles(directory=str(settings.output_dir)), name="static")


@app.get("/")
def root() -> dict:
    """Default root endpoint."""
    return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}
