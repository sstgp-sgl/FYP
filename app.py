#!/usr/bin/env python3
"""Start the Web application.

Default invocation keeps the original photo-to-mesh workflow (SAM2 /
Zero123++ / VGGT environment required). ``python app.py --tour-only`` starts
the CPU-only scene tour server without importing any GPU/model library.
"""
import sys


def main():
    if "--tour-only" in sys.argv[1:]:
        from src.tour.app import tour_main

        tour_main([arg for arg in sys.argv[1:] if arg != "--tour-only"])
    else:
        from src.server import main

        main()


if __name__ == "__main__":
    main()
