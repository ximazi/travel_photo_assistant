"""Render an asymmetric, non-human CameraX calibration target."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw

out = Path(sys.argv[1])
image = Image.new("RGB", (640, 480), "#f4f4f4")
draw = ImageDraw.Draw(image)
draw.ellipse((111, 63, 159, 111), fill="#be1616")
draw.rectangle((440, 185, 519, 243), fill="#164fbc")
draw.polygon([(103, 369), (175, 369), (136, 313)], fill="#168255")
draw.text((192, 18), "UP / LEFT - R", fill="black")
image.save(out)
