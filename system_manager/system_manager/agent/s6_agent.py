"""FastAPI application for s6-overlay agent service management API."""

import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from system_manager.agent.models import (
    ErrorResponse,
    ServiceActionRequest,
    ServiceControlResponse,
    ServiceListResponse,
    ServiceStatus,
)
from system_manager.agent.s6_client import control_service, get_service_status, list_services

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="s6 Agent API",
    description="""
    REST API for managing s6-overlay services within a container.

    This agent provides endpoints to list, check status, and control
    s6-overlay services running in the container.
    """,
    version="0.1.0",
    docs_url=None,  # Disable docs for security (Unix socket only)
    redoc_url=None,
    openapi_url=None,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail or "Unknown error").model_dump(),
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {"message": "s6 Agent API", "version": "0.1.0"}


@app.get(
    "/services",
    response_model=ServiceListResponse,
    tags=["services"],
    summary="List all available services",
    description="Get a list of all s6-overlay services available in this container",
)
async def get_services() -> ServiceListResponse:
    """Get list of all available s6 services.

    Returns:
        ServiceListResponse containing a list of service names.

    Raises:
        HTTPException: 500 if service listing fails.
    """
    try:
        services = list_services()
        return ServiceListResponse(services=services)
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list services: {str(e)}",
        )


@app.get(
    "/services/{name}/status",
    response_model=ServiceStatus,
    tags=["services"],
    summary="Get service status",
    description="Get detailed status information for a specific service",
)
async def get_service_status_endpoint(name: str) -> ServiceStatus:
    """Get status of a specific service.

    Args:
        name: Service name.

    Returns:
        ServiceStatus with detailed service information.

    Raises:
        HTTPException: 404 if service not found, 500 on other errors.
    """
    try:
        return get_service_status(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{name}' not found",
        )
    except Exception as e:
        logger.error(f"Failed to get status for service '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service status: {str(e)}",
        )


@app.post(
    "/services/{name}",
    response_model=ServiceControlResponse,
    tags=["services"],
    summary="Control a service",
    description="Start, stop, or restart a service",
)
async def control_service_endpoint(
    name: str, request: ServiceActionRequest
) -> ServiceControlResponse:
    """Control a service (start, stop, or restart).

    Args:
        name: Service name.
        request: ServiceActionRequest with action to perform.

    Returns:
        ServiceControlResponse confirming the action.

    Raises:
        HTTPException: 404 if service not found, 400 if action invalid, 500 on other errors.
    """
    try:
        control_service(name, request.action)
        return ServiceControlResponse(name=name, action=request.action, result="ok")
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{name}' not found",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to control service '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to control service: {str(e)}",
        )


@app.get(
    "/services/{name}/logs",
    tags=["services"],
    summary="Get service logs (stub)",
    description="Get logs for a service (placeholder for future implementation)",
)
async def get_service_logs(name: str):
    """Get logs for a service (stub implementation).

    Args:
        name: Service name.

    Returns:
        Placeholder response.

    Note:
        This is a stub endpoint. Future implementation might map to
        something like `tail -n 100 /var/log/ros/<service>.log`.
    """
    # TODO: Implement log retrieval
    # For now, return a placeholder
    return {
        "service": name,
        "message": "Log retrieval not yet implemented",
        "note": "Future implementation might map to /var/log/ros/<service>.log",
    }

