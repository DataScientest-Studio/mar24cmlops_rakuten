import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from api.utils.resolve_path import resolve_path
import numpy as np
from io import BytesIO
import pickle
from tensorflow.keras.models import load_model


class tf_trimodel:
    """
    Base class for a tri-model involving text and image processing using LSTM and VGG16.

    Attributes:
        model_type (str): Type of the model (production or staging).
        version (str): Version of the model.
        model_name (str): Name of the model.
        tokenizer (Tokenizer): Tokenizer for text processing.
        lstm (Model): LSTM model for text prediction.
        vgg16 (Model): VGG16 model for image prediction.
        combined_model (Model): Combined model for text and image prediction.
        modalite_mapping (dict): Mapper for prediction classes.
        inv_modalite_mapping (dict): Inverse mapper for prediction classes.
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
        self.model_name = model_name or type(self).__name__
        self.tokenizer = None
        self.lstm = None
        self.vgg16 = None
        self.combined_model = None
        self.modalite_mapping = None
        self.inv_modalite_mapping = None
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
        return resolve_path(f"models/{folder}/{self.model_name}/{self.version}/")

    def load_txt_utils(self):
        """
        Load text utilities including tokenizer and LSTM model.
        """
        model_path = self.get_model_path()
        with open(os.path.join(model_path, "tokenizer.pkl"), "rb") as handle:
            self.tokenizer = pickle.load(handle)
        self.lstm = load_model(os.path.join(model_path, "text_model.keras"))

    def load_img_utils(self):
        """
        Load image utilities including VGG16 model.
        """
        model_path = self.get_model_path()
        self.vgg16 = load_model(os.path.join(model_path, "image_model.keras"))

    def load_model_utils(self):
        """
        Load model utilities including mapper and combined model.
        """
        model_path = self.get_model_path()
        with open(os.path.join(model_path, "modalite_mapping.pkl"), "rb") as handle:
            self.modalite_mapping = pickle.load(handle)
        self.inv_modalite_mapping = {
            value: key for key, value in self.modalite_mapping.items()
        }
        self.combined_model = load_model(
            os.path.join(model_path, "combined_model.keras")
        )

    def path_to_img(self, img_path):
        """
        Load an image from a file path.

        Args:
            img_path (str): Path to the image file.

        Returns:
            np.array: Preprocessed image array.
        """
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        return tf.keras.applications.vgg16.preprocess_input(
            np.expand_dims(img_array, axis=0)
        )

    def byte_to_img(self, file):
        """
        Load an image from a byte stream.

        Args:
            file (bytes): Byte stream of the image.

        Returns:
            np.array: Preprocessed image array.
        """
        img = load_img(BytesIO(file), target_size=(224, 224))
        img_array = img_to_array(img)
        return tf.keras.applications.vgg16.preprocess_input(
            np.expand_dims(img_array, axis=0)
        )

    def _predict(self, text_designation, text_description, image):
        # Prétraitement du texte
        text = str(text_description) + " " + str(text_designation)
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100)

        # Chargement et prétraitement de l'image
        if isinstance(image, str):  # Si l'image est un chemin
            img_array = self.path_to_img(image)
        elif isinstance(image, bytes):  # Si l'image est en bytes
            img_array = self.byte_to_img(image)
        else:
            raise ValueError("Image must be a file path or bytes.")

        # Prédiction avec les modèles
        text_pred = self.lstm.predict(padded_sequence)
        image_pred = self.vgg16.predict(img_array)
        combined_pred = self.combined_model.predict([padded_sequence, img_array])

        # Décoder les prédictions
        text_pred_class = np.argmax(text_pred, axis=1)[0]
        image_pred_class = np.argmax(image_pred, axis=1)[0]
        combined_pred_class = np.argmax(combined_pred, axis=1)[0]

        return {
            "text_prediction": {
                "class": text_pred_class,
                "probability": float(text_pred[0][text_pred_class]),
            },
            "image_prediction": {
                "class": image_pred_class,
                "probability": float(image_pred[0][image_pred_class]),
            },
            "combined_prediction": {
                "class": combined_pred_class,
                "probability": float(combined_pred[0][combined_pred_class]),
            },
        }

    def _predict_from_dataframe(self, df):
        predictions = []

        for index, row in df.iterrows():
            text_designation = row["designation"]
            text_description = row["description"]
            image_path = row["image_path"]

            prediction_result = self._predict(
                text_designation, text_description, image_path
            )
            predictions.append(
                {
                    "text_designation": text_designation,
                    "text_description": text_description,
                    "image_path": image_path,
                    "predictions": prediction_result,
                }
            )

        return predictions

    def predict(self, text_designation, text_description, image):
        result = self._predict(text_designation, text_description, image)
        combined_class = result["combined_prediction"]["class"]
        mapped_class = self.inv_modalite_mapping[combined_class]
        return mapped_class


# Exemple d'utilisation

# model = tf_trimodel("tf_trimodel", "20240708_19-15-54", "production")

# text_designation = "Jeu video"
# text_description = "Titi et les bijoux magiques jeux video enfants gameboy advance"
# image_path = resolve_path("data/images/image_train/image_528113_product_923222.jpg")

# # Prédiction avec un chemin d'image
# result = model.predict(text_designation, text_description, image_path)
# print(result)
