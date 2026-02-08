import os
import re

def extract_text_from_image(image_path):
    """Extract text from an image using OCR (pytesseract + OpenCV)"""
    # Lazy imports to save memory on cloud deploy
    import cv2
    import pytesseract

    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"OpenCV cannot read this image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
    return text.strip()


def pdf_to_images(pdf_path):
    """Convert PDF pages to images using PyMuPDF"""
    # Lazy imports
    import fitz

    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    images = []
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_path = f"page_{i}.png"
        pix.save(img_path)
        images.append(img_path)

    return images


def clean_ocr_text(text):
    """Clean OCR text to retain only valid math characters"""
    text = text.replace("÷", "/")
    text = text.replace("\n", " ").strip()
    text = re.sub(r'[^0-9xX\+\-\*/=\(\)\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_math_expression(expr):
    """Fix common OCR issues like missing operators"""
    expr = expr.strip()
    expr = re.sub(r'([0-9xX\)])\s+([0-9xX\(])', r'\1 + \2', expr)
    expr = re.sub(r'\s+', ' ', expr)
    return expr


def is_equation(text):
    """Detect if the OCR text looks like a solvable equation"""
    math_chars = "+-*/=xX()"
    return any(c in math_chars for c in text) and any(c.isdigit() for c in text)


def solve_math_from_image(image_path):
    """Main function: OCR + solve math expressions or word problems"""
    # Lazy import
    from richiwin_logic import solve_expression, solve_word_problem

    extracted_text = extract_text_from_image(image_path)
    if not extracted_text:
        return "No readable math found in the image."

    print("OCR RAW:", extracted_text)
    text_lower = extracted_text.lower()

    # ---- Geometry / word problems ----
    numbers = re.findall(r'\d+\.?\d*', extracted_text)

    # Triangle area
    if "triangle" in text_lower and "area" in text_lower:
        if len(numbers) >= 2:
            base, height = map(float, numbers[:2])
            area = 0.5 * base * height
            return f"Triangle area = 0.5 * {base} * {height} = {area}"
        return f"Triangle detected but not enough numbers: {numbers}"

    # Rectangle area
    if "rectangle" in text_lower and "area" in text_lower:
        if len(numbers) >= 2:
            length, width = map(float, numbers[:2])
            area = length * width
            return f"Rectangle area = {length} * {width} = {area}"

    # Pythagoras
    if "triangle" in text_lower and "hypotenuse" in text_lower:
        if len(numbers) >= 2:
            a, b = map(float, numbers[:2])
            hypotenuse = (a**2 + b**2)**0.5
            return f"Hypotenuse = sqrt({a}^2 + {b}^2) = {hypotenuse}"

    # ---- Algebra / general math ----
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
