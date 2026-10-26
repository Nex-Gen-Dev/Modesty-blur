"""PyInstaller entry point. Double-click / run with no arguments = live webcam."""

import sys

from modesty_blur.app import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["run", "--source", "0"]
    main()
