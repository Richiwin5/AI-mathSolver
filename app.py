import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from richiwin_brain import solve_anything
from richiwin_camera_math import solve_math_from_image

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return "Richiwin AI Math Solver is Live 🚀"


@app.route("/solve-text", methods=["POST"])
def solve_text():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]

    print("Received:", question)

    result = solve_anything(question)

    print("Result:", result)

    return jsonify({"solution": result})

@app.route("/solve-file", methods=["POST"])
def solve_file():

    print("HEADERS:", request.headers)
    print("FILES:", request.files)
    print("FORM:", request.form)
    print("DATA:", request.data)

    uploaded_file = request.files.get("file")

    from werkzeug.utils import secure_filename
    from richiwin_camera_math import pdf_to_images

    print("FILES:", request.files)  # Debug

    uploaded_file = request.files.get("file")

    if not uploaded_file:
        return jsonify({"error": "No file sent"}), 400

    filename = secure_filename(uploaded_file.filename)

    # Render-safe temp folder
    UPLOAD_FOLDER = "/tmp"
    path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        uploaded_file.save(path)
    except Exception as e:
        return jsonify({"error": f"Save failed: {str(e)}"}), 500

    # IMAGE FILES
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        result = solve_math_from_image(path)

    # PDF FILES
    elif filename.lower().endswith(".pdf"):

        pages = pdf_to_images(path)

        results = []

        for page in pages:
            res = solve_math_from_image(page)
            results.append(res)

        result = results

    else:
        result = f"Unsupported file: {filename}"

    return jsonify({
        "filename": filename,
        "solution": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print(f"🚀 Server running on port {port}")

    app.run(host="0.0.0.0", port=port)
