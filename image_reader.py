import pytesseract
from PIL import Image

def read_image(path):
    return pytesseract.image_to_string(Image.open(path)).strip()
