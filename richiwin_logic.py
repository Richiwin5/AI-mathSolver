# richiwins_word_logic.py
import sympy as sp
import re
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr

# ----------------------------
# 1️⃣ Helper Functions
# ----------------------------

def clean_input(text):
    text = text.lower()
    text = text.replace("solve", "")
    text = text.replace("what is", "")
    text = text.replace("?", "").strip()
    text = text.replace("^", "**")
    # Add implicit multiplication
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', text)
    return text

def detect_variables(expression):
    variables = set(re.findall(r'[a-zA-Z]', expression))
    symbols = {v: sp.symbols(v) for v in variables}
    return symbols

# ----------------------------
# 2️⃣ NLP → Math Parser
# ----------------------------

def parse_word_problem(text):
    """
    Very basic word problem parser:
    Converts sentences into symbolic equations.
    Examples:
    - 'twice a number plus 3 is 7' → '2*x + 3 = 7'
    - 'a number minus 5 equals 10' → 'x - 5 = 10'
    """
    text = clean_input(text)
    # Replace common words with symbols/operators
    text = text.replace("twice", "2*")
    text = text.replace("thrice", "3*")
    text = text.replace("plus", "+")
    text = text.replace("minus", "-")
    text = text.replace("equals", "=")
    text = text.replace("is", "=")
    text = text.replace("times", "*")
    text = text.replace("divided by", "/")
    
    # Detect variable
    if "number" in text:
        text = text.replace("number", "x")

    return text

# ----------------------------
# 3️⃣ Math Solver Functions
# ----------------------------

def solve_expression(expr_text):
    expr_text = clean_input(expr_text)

    # ✅ SAFETY CHECK (THIS FIX)
    if not expr_text.strip():
        return "Invalid or empty math expression"

    symbols = detect_variables(expr_text)
    steps = []

    if "=" in expr_text:
        lhs, rhs = expr_text.split("=")
        lhs_expr = parse_expr(lhs, local_dict=symbols)
        rhs_expr = parse_expr(rhs, local_dict=symbols)

        steps.append(f"Step 1: Equation: {lhs_expr} = {rhs_expr}")
        eq = lhs_expr - rhs_expr
        steps.append(f"Step 2: Move all terms to one side: {eq} = 0")

        solution = sp.solve(eq, list(symbols.values()))
        steps.append(f"Step 3: Solve: {solution}")

        return "\n".join(steps)

    else:
        expr = parse_expr(expr_text, local_dict=symbols)
        simplified = sp.simplify(expr)

        steps.append(f"Step 1: Original expression: {expr}")
        steps.append(f"Step 2: Simplified: {simplified}")

        return "\n".join(steps)


# ----------------------------
# 4️⃣ Word Problem Solver
# ----------------------------

def solve_word_problem(text):
    equation = parse_word_problem(text)
    print(f"Parsed equation: {equation}")
    return solve_expression(equation)

# ----------------------------
# 5️⃣ Main Loop
# ----------------------------
def main():
    print("Richiwin: Hello! I am your reasoning brain 🧠 (advanced word problem ready)")
    
    while True:
        user_input = input("You: ").strip().lower()

        if user_input in ["exit", "quit"]:
            print("Richiwin: Goodbye!")
            break

        # 🧠 Advanced AI Word Problem Detection
        elif any(word in user_input for word in [
            "twice", "thrice", "number", "equals", "sum", "difference",
            "times", "divided", "find", "what is"
        ]):
            print(advanced_word_problem_solver(user_input))

        # ✏️ Direct equation solving
        elif "=" in user_input or "solve" in user_input:
            print(solve_expression(user_input))

        else:
            print("Richiwin: I am still learning this type of question.")


if __name__ == "__main__":
    main()
