"""
FastAPI Main Application
Industrial Digital Twin Inference API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_CONTACT,
    HOST,
    PORT,
    RELOAD,
    ALLOW_ORIGINS,
    ALLOW_CREDENTIALS,
    ALLOW_METHODS,
    ALLOW_HEADERS
)
from .api import models, ensemble, inference

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=ALLOW_METHODS,
    allow_headers=ALLOW_HEADERS
)

# Include routers
app.include_router(models.router)
app.include_router(ensemble.router)
app.include_router(inference.router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/api/v1/info")
async def api_info():
    """Get API information"""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": "FastAPI service for Industrial Digital Twin inference",
        "endpoints": {
            "models": "/api/v1/models",
            "ensemble": "/api/v1/ensemble",
            "inference": "/api/v1/inference",
            "health": "/api/v1/health"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


def main():
    """Run the application"""
    print("=" * 80)
    print(f"=€ Starting {API_TITLE}")
    print(f"=á Server: http://{HOST}:{PORT}")
    print(f"=Ö Docs: http://{HOST}:{PORT}/docs")
    print("=" * 80)

    uvicorn.run(
        "fastapi_inference.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD
    )


if __name__ == "__main__":
    main()
