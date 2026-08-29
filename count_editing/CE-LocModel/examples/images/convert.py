from PIL import Image
import os

def convert_jpg_to_png(jpg_path, png_path):
    img = Image.open(jpg_path)
    img.save(png_path, "PNG")

# Usage
convert_jpg_to_png('2.jpg', '2.png')