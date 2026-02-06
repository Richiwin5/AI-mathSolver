from math_extract import extract_math
from reasoning_math_solver import solve_math

user_input = "Hello Richiwin, solve 5 + 3"

math_expr = extract_math(user_input)

if math_expr:
    answer = solve_math(math_expr)
    print("Solution:", answer)
else:
    print("No math detected")
