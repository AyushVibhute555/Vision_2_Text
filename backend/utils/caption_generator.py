import os
import pickle
import urllib.request
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# PATHS
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "model", "caption_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")

MAX_LENGTH = 35

# =========================
# MODEL DOWNLOAD (IMPORTANT)
# =========================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1SDaL4d8stam_06WwRZATXEVaIb9h1jb6"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded successfully")

# =========================
# LOAD TOKENIZER
# =========================
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# =========================
# REBUILD MODEL ARCHITECTURE
# =========================
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

inputs2 = Input(shape=(MAX_LENGTH,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder = add([fe2, se3])
decoder = Dense(256, activation="relu")(decoder)
outputs = Dense(vocab_size, activation="softmax")(decoder)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.load_weights(MODEL_PATH)

print("✅ Model loaded successfully")

# =========================
# VGG16 FEATURE EXTRACTOR
# =========================
vgg = VGG16(weights="imagenet")
feature_extractor = Model(
    inputs=vgg.input,
    outputs=vgg.layers[-2].output
)

# =========================
# EXTRACT IMAGE FEATURES
# =========================
def extract_feature(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = feature_extractor.predict(image, verbose=0)
    return feature.flatten()

# =========================
# GENERATE CAPTION
# =========================
def generate_caption(image_path):
    feature = extract_feature(image_path)
    feature = np.expand_dims(feature, axis=0)

    in_text = "startseq"

    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_LENGTH)

        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()

# =========================
# TEST (OPTIONAL)
# =========================
if __name__ == "__main__":
    test_image = os.path.join(BASE_DIR, "..", "dataset", "Flickr8k", "Images", "1000268201_693b08cb0e.jpg")
    caption = generate_caption(test_image)
    print("🖼️ Generated Caption:")
    print(caption)