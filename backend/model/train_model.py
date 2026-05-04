import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# PATHS
# =========================
FEATURES_PATH = "backend/model/image_features.pkl"
TOKENIZER_PATH = "backend/model/tokenizer.pkl"
CAPTION_FILE = "dataset/Flickr8k/captions.txt"
MODEL_OUTPUT = "backend/model/caption_model.h5"

# =========================
# LOAD IMAGE FEATURES
# =========================
with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

# =========================
# LOAD TOKENIZER
# =========================
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 35

print("Vocabulary Size:", vocab_size)

# =========================
# LOAD CAPTIONS
# =========================
def load_captions():
    mapping = {}

    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split(",")
            if len(tokens) < 2:
                continue

            img_id = tokens[0].strip()   # KEEP .jpg
            caption = " ".join(tokens[1:]).lower()
            caption = "startseq " + caption + " endseq"

            if img_id not in mapping:
                mapping[img_id] = []

            mapping[img_id].append(caption)

    return mapping

captions_dict = load_captions()
print("Total images with captions:", len(captions_dict))

# =========================
# DATA GENERATOR
# =========================
def data_generator(captions, features, tokenizer, max_length, vocab_size):
    for img_id, caps in captions.items():
        if img_id not in features:
            continue

        # ✅ FIXED: feature is already (4096,)
        feature = features[img_id]
        feature = np.asarray(feature, dtype="float32")

        for cap in caps:
            seq = tokenizer.texts_to_sequences([cap])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical(out_seq, num_classes=vocab_size)

                yield (
                    (
                        feature,                      # (4096,)
                        in_seq.astype("int32")        # (max_length,)
                    ),
                    out_seq.astype("float32")        # (vocab_size,)
                )


# =========================
# TF.DATA DATASET
# =========================
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(
        captions_dict,
        features,
        tokenizer,
        max_length,
        vocab_size
    ),
    output_signature=(
        (
            tf.TensorSpec(shape=(4096,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32),
    ),
)

BATCH_SIZE = 64
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =========================
# MODEL ARCHITECTURE
# =========================
# Image feature branch
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

# Text sequence branch
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Decoder
decoder = add([fe2, se3])
decoder = Dense(256, activation="relu")(decoder)
outputs = Dense(vocab_size, activation="softmax")(decoder)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam")

model.summary()

# =========================
# TRAINING
# =========================
steps = sum(len(caps) for caps in captions_dict.values())
steps_per_epoch = steps // BATCH_SIZE

checkpoint = ModelCheckpoint(
    MODEL_OUTPUT,
    monitor="loss",
    save_best_only=True,
    verbose=1
)

print("🚀 Training started...")

model.fit(
    dataset,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint]
)

print("✅ Training completed successfully")
