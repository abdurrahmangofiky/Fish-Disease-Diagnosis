import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load assets
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Bag-of-words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Get response
def get_response(intents_list, intents_json):
    tag = intents_list[0][0]
    list_of_intents = intents_json['intents']
    result = ""

    for intent in list_of_intents:
        if intent["tag"] == classes[tag]:
            result = random.choice(intent["responses"])
            break

    return result


print("Chatbot siap! Ketik 'quit' untuk keluar.\n")

while True:
    message = input("Anda : ")
    if message.lower() == "quit":
        break
    
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot  :", res)
