import sympy as sp

def solve_math(expression):
    try:
        return sp.sympify(expression).evalf()
    except:
        try:
            x = sp.symbols('x')
            return sp.solve(expression, x)
        except:
            return "Unable to solve"
