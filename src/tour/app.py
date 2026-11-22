"""Tour app factory and startup. ``mount_tour`` attaches the same router and
pages to any FastAPI app (old server or a fresh tour-only app); the tour-only
entry parses CLI args without importing the old server or any GPU library.
"""
from __future__ import annotations

import argparse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import PROJECT_ROOT

from .config import TourConfig
from .repository import (
    ConflictError,
    InvalidDataError,
    NotFoundError,
    PayloadTooLargeError,
    TourError,
    TourRepository,
)
from .routes import create_router
from .service import TourService

WEB_DIR = PROJECT_ROOT / "web"
TOUR_PAGE_DIR = WEB_DIR / "tour"
VENDOR_DIR = WEB_DIR / "vendor"

ERROR_CODES = {
    NotFoundError: ("not_found", 404),
    ConflictError: ("conflict", 409),
    PayloadTooLargeError: ("payload_too_large", 413),
    InvalidDataError: ("invalid_data", 422),
}


def build_service(config: TourConfig, reconstruction_output_dir=None) -> TourService:
    repository = TourRepository(config.output_root)
    repository.recover_interrupted()
    return TourService(repository, config, reconstruction_output_dir=reconstruction_output_dir)


def register_error_handlers(app: FastAPI):
    @app.exception_handler(TourError)
    async def _tour_error(request: Request, exc: TourError):
        code, status = ERROR_CODES.get(type(exc), ("tour_error", exc.status_code))
        return JSONResponse(
            status_code=status,
            content={
                "error": {"code": code, "message": str(exc), "details": None},
                "request_id": getattr(request.state, "request_id", None),
            },
        )


def add_request_id_middleware(app: FastAPI):
    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def mount_static(app: FastAPI):
    app.mount(
        "/tour/static", StaticFiles(directory=str(TOUR_PAGE_DIR), html=True), name="tour-static"
    )
    app.mount(
        "/vendor/three-r128",
        StaticFiles(directory=str(VENDOR_DIR / "three-r128")),
        name="three-r128",
    )


def mount_tour(app: FastAPI, config: TourConfig = None, reconstruction_output_dir=None):
    """Attach the tour router and static pages to an existing FastAPI app."""
    config = config or TourConfig()
    service = build_service(config, reconstruction_output_dir)
    app.state.tour_service = service
    register_error_handlers(app)
    add_request_id_middleware(app)
    mount_static(app)
    app.include_router(create_router(service))

    @app.get("/tour", include_in_schema=False)
    def tour_page():
        from fastapi.responses import FileResponse

        return FileResponse(TOUR_PAGE_DIR / "index.html")

    # Only register the shutdown hook when the host app has no lifespan of its
    # own (e.g. the old server). The tour-only app shuts the service down in
    # its lifespan; runs left active by any other exit are marked interrupted
    # by ``recover_interrupted`` on the next start.
    if getattr(app.router, "lifespan_context", None) is None:
        @app.on_event("shutdown")
        async def _shutdown_tour():
            await service.shutdown()

    return service


def create_tour_app(config: TourConfig = None, reconstruction_output_dir=None) -> FastAPI:
    config = config or TourConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service = getattr(app.state, "tour_service", None)
        if service is not None:
            await service.shutdown()

    app = FastAPI(title="FYP Scene Tour", lifespan=lifespan)
    mount_tour(app, config, reconstruction_output_dir)
    return app


def tour_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tour-only CPU server (no SAM2/Zero123++/VGGT/torch imports)"
    )
    parser.add_argument("--tour-config", default=None, help="tour yaml (default configs/tour.example.yaml)")
    parser.add_argument("--reconstruction-root", default=None,
                        help="existing reconstruction output dir for /scenes/from-reconstruction")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)

    config = TourConfig(args.tour_config)
    reconstruction_root = args.reconstruction_root
    if reconstruction_root is None:
        # Read the old default.yaml only for the reconstruction output dir;
        # no model paths are loaded or touched.
        import yaml

        try:
            with (PROJECT_ROOT / "configs" / "default.yaml").open() as stream:
                old = yaml.safe_load(stream)
            raw = old.get("output_dir", "outputs/reconstruct")
            path = Path(raw)
            reconstruction_root = str(path if path.is_absolute() else PROJECT_ROOT / path)
        except Exception:  # noqa: BLE001
            reconstruction_root = None

    import uvicorn

    app = create_tour_app(config, reconstruction_root)
    print(f"Tour page:   http://{args.host}:{args.port}/tour")
    print(f"Tour root:   {config.output_root}")
    print(f"Recon root:  {reconstruction_root}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
