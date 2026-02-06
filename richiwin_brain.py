# Richiwin – Brain # 

import sympy as sp
import re
import random
import json
import pickle
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from PIL import Image
import pdfplumber
import pytesseract
from sympy.parsing.sympy_parser import parse_expr
from itertools import combinations, permutations
from statistics import mean, median, mode

# ----------------------------
# 1️⃣ Chatbot / Intent Recognition
# ----------------------------
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("chatbot_richiwin.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return None
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

# ----------------------------
# 2️⃣ Math / Symbolic Brain
# ----------------------------

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

def solve_math_full(expression):
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

# ----------------------------
# 3️⃣ Graphs
# ----------------------------
def plot_expression(expr_str):
    try:
        x = sp.symbols('x')
        expr = sp.sympify(expr_str)
        f = sp.lambdify(x, expr, "numpy")
        X = np.linspace(-10,10,400)
        Y = f(X)
        plt.plot(X,Y)
        plt.title(f"Graph of {expr_str}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Graph error:", e)

# ----------------------------
# 4️⃣ Word Problems / Logic
# ----------------------------
def clean_input(text):
    text = text.lower().replace("solve","").replace("what is","").replace("?","").strip()
    text = text.replace("^","**")
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2')
    text = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2')
    return text

def detect_variables(expression):
    variables = set(re.findall(r'[a-zA-Z]', expression))
    symbols = {v: sp.symbols(v) for v in variables}
    return symbols

def parse_word_problem(text):
    text = clean_input(text)
    text = text.replace("twice","2*").replace("thrice","3*")
    text = text.replace("plus","+").replace("minus","-")
    text = text.replace("equals","=").replace("is","=")
    text = text.replace("times","*").replace("divided by","/")
    if "number" in text: text = text.replace("number","x")
    return text

def solve_expression(expr_text):
    expr_text = clean_input(expr_text)
    symbols = detect_variables(expr_text)
    steps = []

    if "=" in expr_text:
        lhs,rhs = expr_text.split("=")
        lhs_expr = parse_expr(lhs, local_dict=symbols)
        rhs_expr = parse_expr(rhs, local_dict=symbols)
        steps.append(f"Step 1: Equation → {lhs_expr} = {rhs_expr}")
        eq = lhs_expr - rhs_expr
        steps.append(f"Step 2: Move all terms → {eq} = 0")
        solution = sp.solve(eq,list(symbols.values()))
        steps.append(f"Step 3: Solve → {solution}")
        return "\n".join(steps)
    else:
        expr = parse_expr(expr_text, local_dict=symbols)
        simplified = sp.simplify(expr)
        steps.append(f"Step 1: Expression → {expr}")
        steps.append(f"Step 2: Simplified → {simplified}")
        return "\n".join(steps)

def solve_word_problem(text):
    equation = parse_word_problem(text)
    return solve_expression(equation)

# ----------------------------
# 5️⃣ Probability & Statistics
# ----------------------------
def solve_probability(text):
    try:
        if "mean" in text:
            numbers = list(map(float,re.findall(r"[-+]?\d*\.\d+|\d+", text)))
            return f"Mean = {mean(numbers)}"
        if "median" in text:
            numbers = list(map(float,re.findall(r"[-+]?\d*\.\d+|\d+", text)))
            return f"Median = {median(numbers)}"
        if "mode" in text:
            numbers = list(map(float,re.findall(r"[-+]?\d*\.\d+|\d+", text)))
            return f"Mode = {mode(numbers)}"
        if "combination" in text or "choose" in text:
            n,m = map(int,re.findall(r'\d+',text))
            return f"C({n},{m}) = {sp.binomial(n,m)}"
        if "permutation" in text:
            n,m = map(int,re.findall(r'\d+',text))
            return f"P({n},{m}) = {sp.factorial(n)/sp.factorial(n-m)}"
        return None
    except Exception as e:
        return f"Probability error: {e}"

# ----------------------------
# 6️⃣ Geometry
# ----------------------------
def solve_geometry(text):
    try:
        # Area
        if "area" in text:
            nums = list(map(float,re.findall(r"[-+]?\d*\.\d+|\d+", text)))
            if "circle" in text: return f"Area = {sp.pi*nums[0]**2}"
            if "square" in text: return f"Area = {nums[0]**2}"
            if "rectangle" in text: return f"Area = {nums[0]*nums[1]}"
        if "perimeter" in text:
            nums = list(map(float,re.findall(r"[-+]?\d*\.\d+|\d+", text)))
            if "square" in text: return f"Perimeter = {4*nums[0]}"
            if "rectangle" in text: return f"Perimeter = {2*(nums[0]+nums[1])}"
        if "volume" in text:
            nums = list(map(float,re.findall(r"[-+]?\d*\.\d+|\d+", text)))
            if "cube" in text: return f"Volume = {nums[0]**3}"
            if "sphere" in text: return f"Volume = {4/3*sp.pi*nums[0]**3}"
        if "pythagoras" in text:
            nums = list(map(float,re.findall(r"[-+]?\d*\.\d+|\d+", text)))
            return f"Hypotenuse = sqrt({nums[0]**2 + nums[1]**2})"
        return None
    except Exception as e:
        return f"Geometry error: {e}"

# ----------------------------
# 7️⃣ Perception: PDF & Image
# ----------------------------
def extract_text_from_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"PDF extraction error: {e}"

def extract_text_from_image(file_path):
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"Image extraction error: {e}"
# ----------------------------
# 8️⃣ Ultimate Brain Function (API-safe)
# ----------------------------
def solve_anything(user_input):
    user_input = user_input.strip().lower()

    # PDF
    if user_input.startswith("pdf:"):
        file_path = user_input.replace("pdf:", "").strip()
        return extract_text_from_pdf(file_path)

    # Image
    if user_input.startswith("image:"):
        file_path = user_input.replace("image:", "").strip()
        return extract_text_from_image(file_path)

    # Probability / Statistics
    prob_stat = solve_probability(user_input)
    if prob_stat:
        return prob_stat

    # Geometry
    geom = solve_geometry(user_input)
    if geom:
        return geom

    # Math / Calculus
    if user_input.startswith("solve"):
        return solve_math_full(user_input)

    # Word Problems
    if any(word in user_input for word in ["number", "twice", "plus", "minus", "equals"]):
        return solve_word_problem(user_input)

    # Chatbot
    ints = predict_class(user_input)
    res = get_response(ints, intents)
    if res:
        return res

    return "I am still learning this type of question."





# # richiwins_super_brain_v2.py
# import sympy as sp
# import matplotlib.pyplot as plt
# import numpy as np
# import re
# import random
# import json
# import pickle
# import nltk
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from math_extract import extract_math
# from richiwin_logic import solve_word_problem

# # ----------------------------
# # Chatbot / Intent Recognition
# # ----------------------------
# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model("chatbot_richiwin.h5")

# def clean_up_sentence(sentence):
#     return [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(sentence)]

# def bag_of_words(sentence):
#     bag = [0] * len(words)
#     for i, w in enumerate(words):
#         if w in clean_up_sentence(sentence):
#             bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     results = [[i, r] for i, r in enumerate(res) if r > 0.25]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     if not intents_list: return None
#     tag = intents_list[0]['intent']
#     for i in intents_json['intents']:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return None

# # ----------------------------
# # Super Math Solver v2
# # ----------------------------
# def solve_math_full(expression):
#     steps = []
#     try:
#         # Clean & normalize
#         expression = expression.lower()
#         expression = expression.replace("solve", "").replace("what is", "").replace("?", "").strip()
#         expression = expression.replace("^", "**")
#         # Handle implicit multiplication
#         expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
#         expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
#         expression = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expression)

#         # Detect variables
#         variables = set(re.findall(r'[a-zA-Z]', expression))
#         symbols = {v: sp.symbols(v) for v in variables}

#         # --------------------
#         # EQUATIONS
#         # --------------------
#         if "=" in expression:
#             left, right = expression.split("=")
#             lhs, rhs = sp.sympify(left, locals=symbols), sp.sympify(right, locals=symbols)
#             steps.append(f"Step 1: Equation → {lhs} = {rhs}")
#             eq = lhs - rhs
#             steps.append(f"Step 2: Move terms → {eq} = 0")
#             solution = sp.solve(eq, list(symbols.values()))
#             steps.append(f"Step 3: Solve → {solution}")
#             return "\n".join(steps)

#         # --------------------
#         # EXPRESSION SIMPLIFICATION
#         # --------------------
#         expr = sp.sympify(expression, locals=symbols)
#         simplified = sp.simplify(expr)
#         steps.append(f"Step 1: Original → {expr}")
#         steps.append(f"Step 2: Simplified → {simplified}")

#         # --------------------
#         # CALCULUS
#         # --------------------
#         if any(k in expression for k in ["derivative", "integrate", "integral", "limit", "partial"]):
#             for var in symbols.values():
#                 if "derivative" in expression:
#                     deriv = sp.diff(expr, var)
#                     steps.append(f"Step 3: Derivative wrt {var} → {deriv}")
#                 if "integrate" in expression or "integral" in expression:
#                     integ = sp.integrate(expr, var)
#                     steps.append(f"Step 3: Integral wrt {var} → {integ}")
#                 if "limit" in expression:
#                     lim = sp.limit(expr, var, 0)
#                     steps.append(f"Step 3: Limit as {var} → {lim}")
#                 if "partial" in expression:
#                     if len(symbols) > 1:
#                         var2 = list(symbols.values())[1]
#                         partial = sp.diff(expr, var2)
#                         steps.append(f"Step 3: Partial derivative wrt {var2} → {partial}")

#         # --------------------
#         # PROBABILITY / STATISTICS
#         # --------------------
#         if "probability" in expression or "binomial" in expression:
#             try:
#                 nums = list(map(int, re.findall(r'\d+', expression)))
#                 if len(nums) >= 2:
#                     k, n = nums[:2]
#                     p = 0.5
#                     prob = sp.binomial(n, k)*p**k*(1-p)**(n-k)
#                     steps.append(f"Step 3: Probability → {prob}")
#             except:
#                 steps.append("Step 3: Could not calculate probability")

#         # --------------------
#         # GEOMETRY
#         # --------------------
#         if any(w in expression for w in ["area", "volume", "radius", "height", "triangle", "circle"]):
#             try:
#                 geom_expr = re.findall(r'[0-9\*\+\-/\(\) ]+', expression)
#                 if geom_expr:
#                     geom_val = sp.sympify(geom_expr[0])
#                     steps.append(f"Step 3: Geometry calculation → {geom_val}")
#             except:
#                 steps.append("Step 3: Could not compute geometry")

#         return "\n".join(steps)

#     except Exception as e:
#         return f"Math error: {e}"

# # ----------------------------
# # Graph Plotting (2D/3D)
# # ----------------------------
# def plot_expression(expr_str, three_d=False):
#     try:
#         if not three_d:
#             x = sp.symbols('x')
#             expr = sp.sympify(expr_str)
#             f = sp.lambdify(x, expr, "numpy")
#             X = np.linspace(-10,10,400)
#             Y = f(X)
#             plt.plot(X,Y)
#             plt.title(f"Graph of {expr_str}")
#             plt.xlabel("x")
#             plt.ylabel("y")
#             plt.grid(True)
#             plt.show()
#         else:
#             # 3D plot example
#             x, y = sp.symbols('x y')
#             expr = sp.sympify(expr_str)
#             f = sp.lambdify((x, y), expr, "numpy")
#             X = np.linspace(-5,5,100)
#             Y = np.linspace(-5,5,100)
#             X, Y = np.meshgrid(X,Y)
#             Z = f(X,Y)
#             from mpl_toolkits.mplot3d import Axes3D
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.plot_surface(X,Y,Z,cmap='viridis')
#             plt.show()
#     except Exception as e:
#         print("Graph error:", e)

# # ----------------------------
# # Main Super Brain Loop v2
# # ----------------------------
# print("Richiwin: Ultimate Super Brain Online 🧠")

# while True:
#     user_input = input("You: ").strip().lower()
#     if user_input in ["exit","quit"]:
#         print("Richiwin: Goodbye!")
#         break

#     # 1️⃣ Math
#     math_expr = extract_math(user_input)
#     if math_expr:
#         print(solve_math_full(math_expr))
#         if "graph" in user_input:
#             plot_expression(math_expr, three_d="3d" in user_input)
#         continue

#     # 2️⃣ Word/Logic problems
#     if any(word in user_input for word in ["number", "twice", "plus", "minus", "equals", "is"]):
#         print(solve_word_problem(user_input))
#         continue

#     # 3️⃣ Chatbot intent
#     ints = predict_class(user_input)
#     res = get_response(ints, intents)
#     if res:
#         print(f"Richiwin: {res}")
#         continue

#     # 4️⃣ Fallback
#     print("Richiwin: I am still learning this type of question.")
