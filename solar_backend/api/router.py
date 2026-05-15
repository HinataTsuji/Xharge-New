"""Main API router registration."""
from __future__ import annotations

from fastapi import APIRouter

from solar_backend.api.routes.analysis import router as analysis_router
from solar_backend.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(analysis_router)
