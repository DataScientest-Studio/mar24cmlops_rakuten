import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from api.utils.resolve_path import resolve_path
import re
import numpy as np
import json
from io import BytesIO
import pickle


class tf_trimodel:
    def __init__(self, model_name, version, model_type):
        self.model_type = model_type
        self.version = version
        if model_name is None:
            self.model_name = type(
                self
            ).__name__  # Utilise le nom de la classe comme nom par défaut
        else:
            self.model_name = model_name
        self.lemmatizer = None
        self.tokenizer = None
        self.stop_words = None
        self.lstm = None
        self.vgg16 = None
        self.mapper = None
        self.best_weights = None
        self.load_txt_utils()
        self.load_img_utils()
        self.load_model_utils()

    def get_model_path(self):
        # Determine folder type based on model_type
        folder = (
            "production_model" if self.model_type == "production" else "staging_models"
        )

        # Determine version folder
        # if self.version == 'latest':
        #     versions = sorted(os.listdir(resolve_path(f'models/{folder}/{self.model_name}/')))
        #     version = versions[-1]
        # else:
        #     version = self.version

        # Construct base path
        base_path = resolve_path(f"models/{folder}/{self.model_name}/{self.version}/")
        return base_path

    def load_txt_utils(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))

        model_path = self.get_model_path()
        with open(
            os.path.join(model_path, "tokenizer_config.json"), "r", encoding="utf-8"
        ) as json_file:
            tokenizer_config = json_file.read()
        self.tokenizer = tokenizer_from_json(tokenizer_config)
        self.lstm = keras.models.load_model(
            os.path.join(model_path, "best_lstm_model.keras")
        )

    def load_img_utils(self):
        model_path = self.get_model_path()
        self.vgg16 = keras.models.load_model(
            os.path.join(model_path, "best_vgg16_model.keras")
        )

    def load_model_utils(self):
        model_path = self.get_model_path()

        with open(os.path.join(model_path, "mapper.json"), "r") as json_file:
            self.mapper = json.load(json_file)

        with open(os.path.join(model_path, "best_weights.pkl"), "rb") as file:
            self.best_weights = pickle.load(file)

    def process_txt(self, text):
        if text is not None:
            text = re.sub(r"[^a-zA-Z]", " ", text)
            words = word_tokenize(text.lower())
            filtered_words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words
            ]
            textes = " ".join(filtered_words[:10])
            sequences = self.tokenizer.texts_to_sequences([textes])
            padded_sequences = pad_sequences(
                sequences, maxlen=10, padding="post", truncating="post"
            )
            return padded_sequences
        return None

    def path_to_img(self, img_path):
        img = load_img(img_path, target_size=(224, 224, 3))
        return img

    def byte_to_img(self, file):
        img = load_img(BytesIO(file), target_size=(224, 224, 3))
        return img

    def process_img(self, img):
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict_text(self, text_sequence):
        probability = self.lstm.predict([text_sequence], verbose=0)
        pred = np.argmax(probability)
        return int(self.mapper[str(pred)]), probability

    def predict_img(self, img_array):
        images = tf.convert_to_tensor([img_array], dtype=tf.float32)
        probability = self.vgg16.predict([images], verbose=0)
        pred = np.argmax(probability)
        return int(self.mapper[str(pred)]), probability

    def agg_prediction(self, txt_prob, img_prob):
        concatenate_proba = (
            self.best_weights[0] * txt_prob + self.best_weights[1] * img_prob
        )
        pred = np.argmax(concatenate_proba)
        return int(self.mapper[str(pred)])

    def predict(self, designation, image):
        text_sequence = self.process_txt(designation)
        if isinstance(image, str):  # If image is a path
            img = self.path_to_img(image)
        elif isinstance(image, bytes):  # If image is bytes
            img = self.byte_to_img(image)
        else:
            raise ValueError("Image must be a file path or bytes.")

        img_array = self.process_img(img)

        _, txt_prob = self.predict_text(text_sequence)
        _, img_prob = self.predict_img(img_array)

        agg_pred = self.agg_prediction(txt_prob, img_prob)
        return agg_pred


# Utilisation de la classe Model

# model = tf_trimodel(model_type='production', version='latest')
# prediction = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg'))
# print(prediction)

# prediction_from_bytes = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', image_bytes)
# print(prediction_from_bytes)
