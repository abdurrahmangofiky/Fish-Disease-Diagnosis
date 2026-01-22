from flask import Flask, render_template, request
from assist_ai import analyze_chat, predict_image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis = None
    chat_response = None
    image_path = None

    if request.method == 'POST':

        # ------- CHATBOT TEXT --------
        if "user_text" in request.form:
            user_text = request.form['user_text']
            chat_response = analyze_chat(user_text)

        # ------- IMAGE UPLOAD --------
        if "image" in request.files:
            file = request.files['image']
            if file.filename != "":
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)
                image_label, acc = predict_image(path)
                diagnosis = f"Penyakit: {image_label} (Akurasi: {acc:.2f}%)"
                image_path = path

    return render_template(
        "index.html",
        diagnosis=diagnosis,
        chat_response=chat_response,
        image_path=image_path
    )


if __name__ == '__main__':
    app.run(debug=True)
