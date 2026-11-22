"""Tour configuration: YAML file plus environment overrides (no secrets logged)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

DEFAULT_TOUR_CONFIG = PROJECT_ROOT / "configs" / "tour.example.yaml"

# Environment overrides. Base URL / model / key stay on the server side; the
# browser can never point the backend at an arbitrary endpoint.
_ENV_OVERRIDES = {
    "tour.output_root": "FYP_TOUR_OUTPUT_ROOT",
    "tour.max_upload_bytes": "FYP_TOUR_MAX_UPLOAD_BYTES",
    "tour.max_points": "FYP_TOUR_MAX_POINTS",
    "tour.preview_points": "FYP_TOUR_PREVIEW_POINTS",
    "tour.language": "FYP_TOUR_LANGUAGE",
    "llm.enabled": "FYP_LLM_ENABLED",
    "llm.base_url": "FYP_LLM_BASE_URL",
    "llm.model": "FYP_LLM_MODEL",
    "llm.api_key_env": "FYP_LLM_API_KEY_ENV",
    "llm.response_mode": "FYP_LLM_RESPONSE_MODE",
    "llm.request_timeout_seconds": "FYP_LLM_REQUEST_TIMEOUT",
    "llm.run_timeout_seconds": "FYP_LLM_RUN_TIMEOUT",
    "llm.max_calls_per_run": "FYP_LLM_MAX_CALLS",
    "llm.max_output_tokens": "FYP_LLM_MAX_OUTPUT_TOKENS",
    "llm.temperature": "FYP_LLM_TEMPERATURE",
}

_TRUE_VALUES = {"1", "true", "yes", "on"}


class TourConfig:
    """Resolved tour configuration.

    ``output_root`` is resolved against the project root so it never depends
    on the process working directory. ``llm_ready()`` reports whether a real
    LLM is configured without exposing the key.
    """

    def __init__(self, config_path=None):
        path = Path(config_path) if config_path else DEFAULT_TOUR_CONFIG
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.config_path = path
        with path.open() as stream:
            self.data = yaml.safe_load(stream) or {}
        self._apply_env()
        self._resolve_paths()
        self._validate()

    # ------------------------------------------------------------- internals
    def _apply_env(self):
        for dotted, env in _ENV_OVERRIDES.items():
            raw = os.environ.get(env)
            if raw is None or raw == "":
                continue
            block, key = dotted.split(".", 1)
            target = self.data.setdefault(block, {})
            if key == "enabled":
                target[key] = raw.strip().lower() in _TRUE_VALUES
            elif key in ("max_upload_bytes", "max_points", "preview_points",
                         "request_timeout_seconds", "run_timeout_seconds",
                         "max_calls_per_run", "max_output_tokens"):
                target[key] = int(raw)
            elif key == "temperature":
                target[key] = float(raw)
            else:
                target[key] = raw

    def _resolve_paths(self):
        raw = self.data.get("tour", {}).get("output_root", "outputs/tours")
        path = Path(str(raw))
        self.output_root = path if path.is_absolute() else PROJECT_ROOT / path

    def _validate(self):
        tour = self.data.setdefault("tour", {})
        if not isinstance(self.output_root, Path):
            raise ValueError("invalid tour.output_root")
        for key in ("max_upload_bytes", "max_points", "preview_points"):
            value = tour[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"invalid tour.{key}")
        if tour["preview_points"] > tour["max_points"]:
            raise ValueError("preview_points must not exceed max_points")
        if not 0.1 <= tour["aspect_ratio"] <= 10:
            raise ValueError("invalid tour.aspect_ratio")
        llm = self.data.setdefault("llm", {})
        llm.setdefault("provider", "chat_completions_compatible")
        llm.setdefault("api_key_env", "FYP_LLM_API_KEY")
        llm.setdefault("response_mode", "text_json")
        llm.setdefault("request_timeout_seconds", 45)
        llm.setdefault("run_timeout_seconds", 100)
        llm.setdefault("max_calls_per_run", 2)
        llm.setdefault("max_output_tokens", 1500)
        llm.setdefault("temperature", 0.2)
        llm.setdefault("enabled", False)
        if llm["response_mode"] not in ("text_json", "json_object"):
            raise ValueError("llm.response_mode must be text_json or json_object")
        if llm["max_calls_per_run"] < 1 or llm["max_calls_per_run"] > 3:
            raise ValueError("llm.max_calls_per_run must be 1..3")
        if type(llm["temperature"]) not in (int, float) or not (0 <= llm["temperature"] <= 2):
            raise ValueError("invalid llm.temperature")

    # ------------------------------------------------------------- accessors
    @property
    def tour(self) -> dict:
        return self.data["tour"]

    @property
    def llm(self) -> dict:
        return self.data["llm"]

    def llm_key(self) -> str | None:
        """API key value if the configured environment variable is set."""
        return os.environ.get(self.llm["api_key_env"]) or None

    def llm_ready(self) -> bool:
        llm = self.llm
        return bool(
            llm.get("enabled") and llm.get("base_url") and llm.get("model") and self.llm_key()
        )

    def capabilities(self) -> dict:
        llm = self.llm
        return {
            "schema_version": 1,
            "tour": {
                "max_upload_bytes": self.tour["max_upload_bytes"],
                "max_points": self.tour["max_points"],
                "preview_points": self.tour["preview_points"],
                "aspect_ratio": self.tour["aspect_ratio"],
                "language": self.tour["language"],
            },
            "llm": {
                "enabled": bool(llm.get("enabled")),
                "configured": self.llm_ready(),
                "provider": llm.get("provider"),
                "model": llm.get("model") or None,
                # Never include base_url or key material.
            },
        }
