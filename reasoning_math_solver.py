import sympy as sp
import re

def solve_math(expression):
    try:
        steps = []

        # Clean text
        expression = expression.lower()
        expression = expression.replace("solve", "")
        expression = expression.replace("what is", "")
        expression = expression.replace("?", "").strip()

        # Convert ^ to **
        expression = expression.replace("^", "**")

        # Implicit multiplication
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
        expression = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expression)

        # Detect variables
        variables = set(re.findall(r'[a-zA-Z]', expression))
        symbols = {v: sp.symbols(v) for v in variables}

        # --------------------
        # EQUATION CASE
        # --------------------
        if "=" in expression:
            left, right = expression.split("=")

            lhs = sp.sympify(left, locals=symbols)
            rhs = sp.sympify(right, locals=symbols)

            steps.append(f"Step 1: Start with the equation {lhs} = {rhs}")

            eq = lhs - rhs
            steps.append(f"Step 2: Move all terms to one side → {eq} = 0")

            solution = sp.solve(eq, list(symbols.values()))
            steps.append(f"Step 3: Solve for the variable(s) → {solution}")

            return "\n".join(steps)

        # --------------------
        # EXPRESSION CASE
        # --------------------
        expr = sp.sympify(expression, locals=symbols)
        simplified = sp.simplify(expr)

        steps.append(f"Step 1: Original expression → {expr}")
        steps.append(f"Step 2: Simplified expression → {simplified}")

        return "\n".join(steps)

    except Exception as e:
        return f"Math error: {e}"




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
