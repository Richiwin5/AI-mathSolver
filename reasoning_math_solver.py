import sympy as sp
import re
from sympy.parsing.sympy_parser import parse_expr

def clean_input(text):
    text = text.lower().replace("solve", "").replace("what is", "").replace("?", "").strip()
    text = text.replace("^", "**")
    # Implicit multiplication
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)
    re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', text)
    return text

def detect_variables(expression):
    variables = set(re.findall(r'[a-zA-Z]', expression))
    symbols = {v: sp.symbols(v) for v in variables}
    return symbols

def solve_math(expression):
    """
    New step-by-step solver for linear equations
    """
    expression = clean_input(expression)
    if not expression.strip():
        return "Invalid or empty expression"

    symbols = detect_variables(expression)
    steps = []

    # Handle equations
    if "=" in expression:
        lhs, rhs = expression.split("=")
        lhs_expr = parse_expr(lhs, local_dict=symbols)
        rhs_expr = parse_expr(rhs, local_dict=symbols)
        var = list(symbols.values())[0]  # assume one variable

        steps.append(f"Step 1: Start with the equation → {lhs_expr} = {rhs_expr}")

        # Move constants from LHS to RHS
        lhs_terms = lhs_expr.as_ordered_terms()
        const_terms = [t for t in lhs_terms if t.free_symbols != {var}]
        if const_terms:
            const_sum = sum(const_terms)
            new_lhs = lhs_expr - const_sum
            new_rhs = rhs_expr - const_sum
            steps.append(f"Step 2: Subtract {const_sum} from both sides → {new_lhs} = {new_rhs}")
        else:
            new_lhs = lhs_expr
            new_rhs = rhs_expr

        # Divide by coefficient of variable
        coeff = new_lhs.coeff(var)
        if coeff != 1:
            final_rhs = new_rhs / coeff
            steps.append(f"Step 3: Divide both sides by {coeff} → {var} = {final_rhs}")
        else:
            final_rhs = new_rhs
            steps.append(f"Step 3: Variable already isolated → {var} = {final_rhs}")

        # Simplify numerical result
        simplified = sp.simplify(final_rhs)
        steps.append(f"Step 4: Simplify → {var} = {simplified}")

        return "\n".join(steps)

    # Handle simple expressions
    else:
        expr = parse_expr(expression, local_dict=symbols)
        simplified = sp.simplify(expr)
        steps.append(f"Step 1: Original expression → {expr}")
        steps.append(f"Step 2: Simplified → {simplified}")
        return "\n".join(steps)




# import sympy as sp
# import re

# def solve_math(expression):
#     try:
#         steps = []

#         # Clean text
#         expression = expression.lower()
#         expression = expression.replace("solve", "")
#         expression = expression.replace("what is", "")
#         expression = expression.replace("?", "").strip()

#         # Convert ^ to **
#         expression = expression.replace("^", "**")

#         # Implicit multiplication
#         expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
#         expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
#         expression = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expression)

#         # Detect variables
#         variables = set(re.findall(r'[a-zA-Z]', expression))
#         symbols = {v: sp.symbols(v) for v in variables}

#         # --------------------
#         # EQUATION CASE
#         # --------------------
#         if "=" in expression:
#             left, right = expression.split("=")

#             lhs = sp.sympify(left, locals=symbols)
#             rhs = sp.sympify(right, locals=symbols)

#             steps.append(f"Step 1: Start with the equation {lhs} = {rhs}")

#             eq = lhs - rhs
#             steps.append(f"Step 2: Move all terms to one side → {eq} = 0")

#             solution = sp.solve(eq, list(symbols.values()))
#             steps.append(f"Step 3: Solve for the variable(s) → {solution}")

#             return "\n".join(steps)

#         # --------------------
#         # EXPRESSION CASE
#         # --------------------
#         expr = sp.sympify(expression, locals=symbols)
#         simplified = sp.simplify(expr)

#         steps.append(f"Step 1: Original expression → {expr}")
#         steps.append(f"Step 2: Simplified expression → {simplified}")

#         return "\n".join(steps)

#     except Exception as e:
#         return f"Math error: {e}"




# import sympy as sp
# import re

# def solve_math(expression):
#     try:
#         steps = []

#         # Clean text
#         expression = expression.lower()
#         expression = expression.replace("solve", "")
#         expression = expression.replace("what is", "")
#         expression = expression.replace("?", "").strip()

#         # Convert ^ to **
#         expression = expression.replace("^", "**")

#         # Implicit multiplication
#         expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
#         expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
#         expression = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expression)

#         # Detect variables
#         variables = set(re.findall(r'[a-zA-Z]', expression))
#         symbols = {v: sp.symbols(v) for v in variables}

#         # --------------------
#         # EQUATION CASE
#         # --------------------
#         if "=" in expression:
#             left, right = expression.split("=")

#             lhs = sp.sympify(left, locals=symbols)
#             rhs = sp.sympify(right, locals=symbols)

#             steps.append(f"Step 1: Start with the equation {lhs} = {rhs}")

#             # Move all terms to one side
#             eq = lhs - rhs
#             steps.append(f"Step 2: Move all terms to one side → {eq} = 0")

#             # Solve equation
#             solution = sp.solve(eq)

#             steps.append(f"Step 3: Solve for the variable → {solution}")

#             return "\n".join(steps)

#         # --------------------
#         # EXPRESSION CASE
#         # --------------------
#         expr = sp.sympify(expression, locals=symbols)
#         simplified = sp.simplify(expr)

#         steps.append(f"Step 1: Original expression → {expr}")
#         steps.append(f"Step 2: Simplified expression → {simplified}")

#         return "\n".join(steps)

#     except Exception as e:
#         return f"Math error: {e}"
