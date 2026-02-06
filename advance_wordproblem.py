# ================================
# 9️⃣ Advanced Word Problem AI Parsing
# ================================
import spacy
nlp = spacy.load("en_core_web_sm")  # English NLP model

def advanced_word_problem_solver(sentence):
    """
    Converts a natural English math problem into an equation
    and solves it step by step.
    """
    try:
        steps = []

        # 1️⃣ Preprocessing
        sentence_clean = sentence.lower().replace("?", "").replace(",", "")
        sentence_clean = sentence_clean.replace("what is", "").replace("find", "")

        # 2️⃣ Use NLP to detect numbers and variables
        doc = nlp(sentence_clean)
        numbers = [token.text for token in doc if token.like_num]
        vars_found = [token.text for token in doc if token.is_alpha and token.text not in ["solve","number","times","plus","minus","equals","is"]]
        
        # Assume first variable is 'x' if none
        var_symbol = sp.symbols(vars_found[0]) if vars_found else sp.symbols('x')

        # 3️⃣ Replace word patterns with symbols/operators
        sentence_clean = sentence_clean.replace("twice", "2*")
        sentence_clean = sentence_clean.replace("thrice", "3*")
        sentence_clean = sentence_clean.replace("plus", "+")
        sentence_clean = sentence_clean.replace("minus", "-")
        sentence_clean = sentence_clean.replace("times", "*")
        sentence_clean = sentence_clean.replace("divided by", "/")
        sentence_clean = sentence_clean.replace("equals","=").replace("is","=")

        # Replace generic 'number' with variable
        sentence_clean = re.sub(r'\bnumber\b', str(var_symbol), sentence_clean)

        # Extract valid math characters only
        equation_chars = re.findall(r'[0-9\*\+\-/\^\(\)'+str(var_symbol)+r'=]+', sentence_clean)
        if not equation_chars:
            return "Richiwin: Could not parse the word problem."

        eq_str = "".join(equation_chars)

        # 4️⃣ Solve equation
        if "=" in eq_str:
            lhs,rhs = eq_str.split("=")
            lhs_expr = sp.sympify(lhs)
            rhs_expr = sp.sympify(rhs)
            eq = lhs_expr - rhs_expr
            steps.append(f"Step 1: Convert sentence → {lhs_expr} = {rhs_expr}")
            steps.append(f"Step 2: Move all terms → {eq} = 0")
            solution = sp.solve(eq, var_symbol)
            steps.append(f"Step 3: Solve for {var_symbol} → {solution}")
        else:
            expr = sp.sympify(eq_str)
            steps.append(f"Step 1: Expression → {expr}")
            steps.append(f"Step 2: Simplified → {sp.simplify(expr)}")

        return "\n".join(steps)
    
    except Exception as e:
        return f"Richiwin: Could not solve the word problem ({e})"
