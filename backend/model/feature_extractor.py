import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Paths
IMAGE_FOLDER = "dataset/Flickr8k/Images"
OUTPUT_PATH = "backend/model/image_features.pkl"

# Load pretrained VGG16 model
base_model = VGG16(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features():
    features = {}

    for img_name in tqdm(os.listdir(IMAGE_FOLDER)):
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        # Load & preprocess image
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Extract features
        feature = model.predict(image, verbose=0)
        features[img_name] = feature.flatten()

    return features

if __name__ == "__main__":
    print("🔍 Extracting image features...")
    features = extract_features()

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(features, f)

    print(f"✅ Features saved to {OUTPUT_PATH}")
    print(f"📦 Total images processed: {len(features)}")
