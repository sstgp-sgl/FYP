"""Tour module. No I/O or model loading at import time.

Importing this package must not import torch, fcntl, sharp_cli, SAM2,
Zero123++, VGGT or any GPU library. The tour-only startup path depends on
this guarantee.
"""
