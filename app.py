from flask import Flask, request, jsonify
from flask_cors import CORS
from richiwin_brain import solve_anything
from richiwin_camera_math import solve_math_from_image

app = Flask(__name__)
CORS(app)

@app.route("/solve-text", methods=["POST"])
def solve_text():
    data = request.get_json(force=True)
    print("Received request:", data)  #  debug
    question = data.get("question", "")
    result = solve_anything(question)
    print("Solution:", result)  #  debug
    return jsonify({"solution": result})

@app.route("/solve-file", methods=["POST"])
def solve_file():
    from werkzeug.utils import secure_filename
    import os

    uploaded_file = request.files.get("file")  # key = 'file'

    if not uploaded_file:
        return jsonify({"error": "No file sent"}), 400

    # Secure filename
    filename = secure_filename(uploaded_file.filename)
    path = os.path.join(os.getcwd(), filename)
    uploaded_file.save(path)

    # Determine processing based on extension
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        result = solve_math_from_image(path)
    elif filename.lower().endswith(".pdf"):
        result = "PDF processing not implemented yet"
    elif filename.lower().endswith((".doc", ".docx")):
        result = "Word processing not implemented yet"
    else:
        result = f"Cannot process {filename}"

    return jsonify({"filename": filename, "solution": result})



if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000
    print(f"Richiwin: Math solver backend is ready!")
    app.run(host=host, port=port, debug=True)

  
