# =============================
# Richiwin – Camera Math OCR
# =============================
import os
import re
import shutil
import cv2
import numpy as np
from PIL import Image
import pytesseract

# ----------------------------
# 0️⃣ Tesseract Setup
# ----------------------------
tesseract_path = shutil.which("tesseract")
if tesseract_path is None:
    print("⚠️ Warning: Tesseract OCR not found. Image OCR will not work on this server.")
else:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ----------------------------
# 1️⃣ Fix Implicit Multiplication
# ----------------------------
def fix_implicit_multiplication(expr: str) -> str:
    expr = expr.replace(" ", "")
    expr = re.sub(r'(\d)\(', r'\1*(', expr)
    expr = re.sub(r'\)(\d)', r')*\1', expr)
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'\)([a-zA-Z])', r')*\1', expr)
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
    expr = re.sub(r'\)\(', r')*(', expr)
    return expr

# ----------------------------
# 2️⃣ Clean Input for Math
# ----------------------------
def clean_input(text: str) -> str:
    text = text.lower().replace("solve", "").replace("what is", "").replace("?", "").strip()
    text = text.replace("^", "**")
    text = fix_implicit_multiplication(text)
    return text

# ----------------------------
# 3️⃣ Extract Text from Image
# ----------------------------
def extract_text_from_image(file_path: str) -> str:
    if tesseract_path is None:
        return "Tesseract OCR not installed. Cannot extract text from images."

    if not os.path.exists(file_path):
        return f"Image file not found: {file_path}"

    try:
        img = cv2.imread(file_path)
        if img is None:
            return f"Cannot read image: {file_path}"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        pil_img = Image.fromarray(gray)
        text = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 6")
        return text.strip()
    except Exception as e:
        return f"Image extraction error: {e}"

# ----------------------------
# 4️⃣ Clean OCR Text
# ----------------------------
def clean_ocr_text(text: str) -> str:
    text = text.replace("÷", "/")
    text = text.replace("\n", " ").strip()
    text = re.sub(r'[^0-9xX\+\-\*/=\(\)\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_math_expression(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r'([0-9xX\)])\s+([0-9xX\(])', r'\1 + \2', expr)
    expr = re.sub(r'\s+', ' ', expr)
    return expr

def is_equation(text: str) -> bool:
    math_chars = "+-*/=xX()"
    return any(c in math_chars for c in text) and any(c.isdigit() for c in text)

# ----------------------------
# 5️⃣ Solve Math from Image
# ----------------------------
def solve_math_from_image(image_path: str) -> str:
    from richiwin_logic import solve_expression, solve_word_problem

    extracted_text = extract_text_from_image(image_path)
    if not extracted_text or "error" in extracted_text.lower():
        return f"OCR Error or no text detected: {extracted_text}"

    print("OCR RAW:", extracted_text)

    text_lower = extracted_text.lower()
    numbers = re.findall(r'\d+\.?\d*', extracted_text)

    # Geometry: Triangle area
    if "triangle" in text_lower and "area" in text_lower:
        if len(numbers) >= 2:
            base, height = map(float, numbers[:2])
            area = 0.5 * base * height
            return f"Triangle area = 0.5 * {base} * {height} = {area}"
        return f"Triangle detected but not enough numbers: {numbers}"

    # Geometry: Rectangle area
    if "rectangle" in text_lower and "area" in text_lower:
        if len(numbers) >= 2:
            length, width = map(float, numbers[:2])
            area = length * width
            return f"Rectangle area = {length} * {width} = {area}"

    # Pythagoras
    if "triangle" in text_lower and "hypotenuse" in text_lower:
        if len(numbers) >= 2:
            a, b = map(float, numbers[:2])
            hypotenuse = (a**2 + b**2) ** 0.5
            return f"Hypotenuse = sqrt({a}^2 + {b}^2) = {hypotenuse}"

    # Algebra / general math
    cleaned_text = clean_ocr_text(extracted_text)
    normalized_text = normalize_math_expression(cleaned_text)
    print("CLEANED:", cleaned_text)
    print("NORMALIZED:", normalized_text)

    if is_equation(normalized_text):
        try:
            return solve_expression(normalized_text)
        except Exception:
            return "Could not understand the math expression clearly."
    else:
        return f"Extracted numbers (cannot solve automatically): {numbers}"







# import pytesseract
# import cv2
# import re
# from richiwin_logic import solve_expression, solve_word_problem

# def extract_text_from_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Failed to read image file")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )

#     text = pytesseract.image_to_string(gray, config="--psm 6")
#     return text.strip()


# def clean_ocr_text(text):
#     """Keep only math characters to prevent SymPy errors"""
#     text = text.replace("\n", " ").strip()
#     text = re.sub(r'[^0-9xX\+\-\*/=\(\)\s]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text


# def normalize_math_expression(expr):
#     """Fix common OCR issues like missing operators"""
#     expr = expr.strip()
#     expr = re.sub(r'([0-9xX\)])\s+([0-9xX\(])', r'\1 + \2', expr)
#     expr = re.sub(r'\s+', ' ', expr)
#     return expr


# def is_equation(text):
#     """Detect if the OCR text looks like a solvable equation"""
#     math_chars = "+-*/=xX()"
#     return any(c in math_chars for c in text) and any(c.isdigit() for c in text)

# def solve_math_from_image(image_path):
#     # 1️⃣ Extract text from image
#     extracted_text = extract_text_from_image(image_path)
#     if not extracted_text:
#         return "No readable math found in the image."

#     print("OCR RAW:", extracted_text)
#     text_lower = extracted_text.lower()

#     # 2️⃣ Geometry / word-problem detection

#     # ---- TRIANGLE AREA ----
#     if "triangle" in text_lower and "area" in text_lower:
#         numbers = re.findall(r'\d+\.?\d*', extracted_text)
#         if len(numbers) >= 2:
#             base, height = map(float, numbers[:2])
#             area = 0.5 * base * height
#             return f"Triangle area = 0.5 * {base} * {height} = {area}"
#         else:
#             return f"Triangle detected but not enough numbers: {numbers}"

#     # ---- RECTANGLE AREA ----
#     if "rectangle" in text_lower and "area" in text_lower:
#         numbers = re.findall(r'\d+\.?\d*', extracted_text)
#         if len(numbers) >= 2:
#             length, width = map(float, numbers[:2])
#             area = length * width
#             return f"Rectangle area = {length} * {width} = {area}"

#     # ---- PYTHAGORAS ----
#     if "triangle" in text_lower and "hypotenuse" in text_lower:
#         numbers = re.findall(r'\d+\.?\d*', extracted_text)
#         if len(numbers) >= 2:
#             a, b = map(float, numbers[:2])
#             hypotenuse = (a**2 + b**2)**0.5
#             return f"Hypotenuse = sqrt({a}^2 + {b}^2) = {hypotenuse}"

#     # 3️⃣ If it’s algebra / general math
#     cleaned_text = clean_ocr_text(extracted_text)
#     normalized_text = normalize_math_expression(cleaned_text)

#     print("CLEANED:", cleaned_text)
#     print("NORMALIZED:", normalized_text)

#     if is_equation(normalized_text):
#         try:
#             return solve_expression(normalized_text)
#         except Exception as e:
#             return f"Could not solve expression: {e}"
#     else:
#         # fallback for numbers or descriptive text
#         numbers = re.findall(r'\d+\.?\d*', extracted_text)
#         return f"Extracted numbers (cannot solve automatically): {numbers}"
