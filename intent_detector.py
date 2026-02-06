import re

def detect_intent(text):
    math_keywords = ['solve', 'find', 'calculate', 'x', '=', '+', '-', '*', '/']
    
    for k in math_keywords:
        if k in text.lower():
            return "math"
    
    return "general"
