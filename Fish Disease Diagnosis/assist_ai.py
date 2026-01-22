import json
import numpy as np
import pickle
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


chat_model = load_model("chatbot_model.h5")

with open("words.pkl", "rb") as f:
    words = pickle.load(f)

with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)



lemmatizer = nltk.WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        if s in words:
            bag[words.index(s)] = 1
    return np.array(bag)


def analyze_chat(msg):
    bow_data = bow(msg, words).reshape(1, -1)
    res = chat_model.predict(bow_data)[0]
    idx = np.argmax(res)
    tag = classes[idx]
    prob = res[idx]

    if prob < 0.40:
        return "Maaf, saya belum memiliki informasi untuk itu."

    
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return intent["responses"][0]

    return "Saya tidak mengerti, coba jelaskan lagi."



img_model = load_model("model_ikan.h5")

with open("label_map.json", "r") as f:
    label_map = json.load(f)



def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = img_model.predict(x)[0]
    idx = int(np.argmax(pred))
    acc = float(np.max(pred))

    label = label_map[str(idx)]
    return label, acc * 100
