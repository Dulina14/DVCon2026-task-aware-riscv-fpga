#!/usr/bin/env python3
"""Convert the board's result.ppm to PNG for viewing on Windows.
    python ppm_to_png.py result.ppm result.png
Windows Photos cannot open PPM; this is purely a container change and is not
part of the pipeline."""
import sys
from PIL import Image
src = sys.argv[1] if len(sys.argv) > 1 else 'result.ppm'
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit('.', 1)[0] + '.png'
Image.open(src).save(dst)
print('wrote', dst)
