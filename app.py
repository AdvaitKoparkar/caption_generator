import os
from PIL import Image
from flask import Flask, request, render_template, jsonify
from models import generate_captions, rank_captions, modify_caption

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        user_prompt = request.form.get("user_prompt", "")

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            image = Image.open(image_path)
            captions = generate_captions(image, user_prompt)
            top_captions = rank_captions(image, captions)

            return jsonify({"captions": top_captions, "image_path": image_path})

    return render_template("index.html")

@app.route("/modify_caption", methods=["POST"])
def modify_caption_api():
    data = request.json
    selected_caption = data.get("selected_caption", "")
    user_instruction = data.get("user_instruction", "")

    if not selected_caption or not user_instruction:
        return jsonify({"error": "Missing caption or instruction"}), 400

    modified_caption = modify_caption(selected_caption, user_instruction)
    return jsonify({"modified_caption": modified_caption})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
