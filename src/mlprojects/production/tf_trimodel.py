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
    """
    Base class for a tri-model involving text and image processing using LSTM and VGG16.
    
    Attributes:
        model_type (str): Type of the model (production or staging).
        version (str): Version of the model.
        model_name (str): Name of the model.
        lemmatizer (WordNetLemmatizer): Lemmatizer for text processing.
        tokenizer (Tokenizer): Tokenizer for text processing.
        stop_words (set): Set of stop words for text processing.
        lstm (Model): LSTM model for text prediction.
        vgg16 (Model): VGG16 model for image prediction.
        mapper (dict): Mapper for prediction classes.
        best_weights (list): Weights for aggregating predictions.
    """

    def __init__(self, model_name, version, model_type):
        """
        Initialize the tf_trimodel class.
        
        Args:
            model_name (str): Name of the model.
            version (str): Version of the model.
            model_type (str): Type of the model (production or staging).
        """
        self.model_type = model_type
        self.version = version
        if model_name is None:
            self.model_name = type(
                self
            ).__name__  # Use the class name as the default name
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
        """
        Get the model path based on model type and version.
        
        Returns:
            str: Path to the model.
        """
        folder = (
            "production_model" if self.model_type == "production" else "staging_models"
        )

        base_path = resolve_path(f"models/{folder}/{self.model_name}/{self.version}/")
        return base_path

    def load_txt_utils(self):
        """
        Load text utilities including tokenizer, stop words, and LSTM model.
        """
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
        """
        Load image utilities including VGG16 model.
        """
        model_path = self.get_model_path()
        self.vgg16 = keras.models.load_model(
            os.path.join(model_path, "best_vgg16_model.keras")
        )

    def load_model_utils(self):
        """
        Load model utilities including mapper and best weights.
        """
        model_path = self.get_model_path()

        with open(os.path.join(model_path, "mapper.json"), "r") as json_file:
            self.mapper = json.load(json_file)

        with open(os.path.join(model_path, "best_weights.pkl"), "rb") as file:
            self.best_weights = pickle.load(file)

    def process_txt(self, text):
        """
        Process the input text for prediction.
        
        Args:
            text (str): Input text.
        
        Returns:
            np.array: Processed text sequence.
        """
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
        """
        Load an image from a file path.
        
        Args:
            img_path (str): Path to the image file.
        
        Returns:
            PIL.Image: Loaded image.
        """
        img = load_img(img_path, target_size=(224, 224, 3))
        return img

    def byte_to_img(self, file):
        """
        Load an image from a byte stream.
        
        Args:
            file (bytes): Byte stream of the image.
        
        Returns:
            PIL.Image: Loaded image.
        """
        img = load_img(BytesIO(file), target_size=(224, 224, 3))
        return img

    def process_img(self, img):
        """
        Process the input image for prediction.
        
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            np.array: Processed image array.
        """
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict_text(self, text_sequence):
        """
        Predict the class of the input text sequence using LSTM model.
        
        Args:
            text_sequence (np.array): Processed text sequence.
        
        Returns:
            tuple: Predicted class and probability.
        """
        probability = self.lstm.predict([text_sequence], verbose=0)
        pred = np.argmax(probability)
        return int(self.mapper[str(pred)]), probability

    def predict_img(self, img_array):
        """
        Predict the class of the input image array using VGG16 model.
        
        Args:
            img_array (np.array): Processed image array.
        
        Returns:
            tuple: Predicted class and probability.
        """
        images = tf.convert_to_tensor([img_array], dtype=tf.float32)
        probability = self.vgg16.predict([images], verbose=0)
        pred = np.argmax(probability)
        return int(self.mapper[str(pred)]), probability

    def agg_prediction(self, txt_prob, img_prob):
        """
        Aggregate predictions from text and image models.
        
        Args:
            txt_prob (np.array): Probability distribution from text model.
            img_prob (np.array): Probability distribution from image model.
        
        Returns:
            int: Aggregated predicted class.
        """
        concatenate_proba = (
            self.best_weights[0] * txt_prob + self.best_weights[1] * img_prob
        )
        pred = np.argmax(concatenate_proba)
        return int(self.mapper[str(pred)])

    def predict(self, designation, image):
        """
        Predict the class based on both text and image inputs.
        
        Args:
            designation (str): Input text.
            image (str or bytes): Input image path or byte stream.
        
        Returns:
            int: Predicted class.
        """
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
