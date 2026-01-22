import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import sys

# Load model
model = tf.keras.models.load_model("model_ikan.h5")

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Balik mapping (0:aeromonas, 1:saprolegnia)
inv_label_map = {v: k for k, v in label_map.items()}

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0

    pred = model.predict(img_arr)
    class_idx = np.argmax(pred)
    class_name = inv_label_map[class_idx]
    
    print(f"Prediksi: {class_name}")
    print(f"Akurasi: {pred[0][class_idx]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.10 predict.py <image_path>")
    else:
        predict(sys.argv[1])
