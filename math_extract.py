import re

def extract_math(text):
    text = text.lower()

    # Remove command words
    text = text.replace("solve", "").strip()

    matches = re.findall(r"[0-9a-zA-Z+\-*/^().= ]+", text)
    if matches:
        return matches[-1].strip()
    return None


# import re

# def extract_math(text):
#     match = re.findall(r"[0-9+\-*/().xX= ]+", text)
#     if match:
#         return match[-1].strip()
#     return None
