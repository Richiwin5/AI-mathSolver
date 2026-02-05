from flask import Flask, request, jsonify
from flask_cors import CORS
from richiwin_brain import solve_anything
from richiwin_camera_math import solve_math_from_image

app = Flask(__name__)
CORS(app)

@app.route("/solve-text", methods=["POST"])
def solve_text():
    data = request.get_json(force=True)
    print("Received request:", data)  # ✅ debug
    question = data.get("question", "")
    result = solve_anything(question)
    print("Solution:", result)  # ✅ debug
    return jsonify({"solution": result})

@app.route("/solve-image", methods=["POST"])
def solve_image():
    print("====== SOLVE IMAGE CALLED ======")
    print("METHOD:", request.method)
    print("CONTENT TYPE:", request.content_type)
    print("FILES:", request.files)

    image = request.files.get("image")

    if not image:
        print("IMAGE NOT FOUND")
        return jsonify({"error": "No image sent"}), 400

    print("✅ IMAGE RECEIVED")
    print("Filename:", image.filename)
    print("Mimetype:", image.mimetype)

    path = "uploaded.jpg"
    image.save(path)
    print("✅ IMAGE SAVED")

    result = solve_math_from_image(path)
    print("Image solution:", result)

    return jsonify({"solution": result})


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000
    print(f"Richiwin: Math solver backend is ready!")
  
