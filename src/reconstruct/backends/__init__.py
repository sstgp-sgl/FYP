"""Reconstruction backends.

Only lightweight backends are imported at package import time; ``vggt`` is
loaded lazily inside :func:`get_backend` so the framework core stays
torch-free.
"""
from __future__ import annotations

from src.reconstruct.backends.base import ReconstructionBackend, ReconstructionResult
from src.reconstruct.backends.dummy import DummyBackend

__all__ = ["ReconstructionBackend", "ReconstructionResult", "DummyBackend", "get_backend"]


def get_backend(name: str, cfg: dict) -> ReconstructionBackend:
    if name == "dummy":
        return DummyBackend(cfg)
    if name == "vggt":
        from src.reconstruct.backends.vggt import VGGTBackend

        return VGGTBackend(cfg)
    raise ValueError(
        f"unknown backend {name!r}; available: dummy, vggt (set cfg['backend'])"
    )
