import string
import pickle

CAPTION_FILE = "dataset/Flickr8k/captions.txt"
TOKENIZER_PATH = "backend/model/tokenizer.pkl"

def load_captions():
    mapping = {}

    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")

            if len(tokens) < 2:
                continue

            image_id, caption = tokens[0], " ".join(tokens[1:])
            image_id = image_id.split(".")[0]

            caption = caption.lower()
            caption = caption.translate(str.maketrans("", "", string.punctuation))
            caption = caption.strip()

            caption = "startseq " + caption + " endseq"

            if image_id not in mapping:
                mapping[image_id] = []

            mapping[image_id].append(caption)

    return mapping

def create_tokenizer(captions_dict):
    all_captions = []

    for captions in captions_dict.values():
        all_captions.extend(captions)

    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    return tokenizer

if __name__ == "__main__":
    print("📖 Loading and preprocessing captions...")
    captions = load_captions()
    tokenizer = create_tokenizer(captions)

    print("✅ Tokenizer created and saved")
    print("📊 Vocabulary size:", len(tokenizer.word_index) + 1)
