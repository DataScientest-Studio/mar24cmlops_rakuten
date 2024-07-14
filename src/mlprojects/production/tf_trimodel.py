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
        self.lstm = None
        self.vgg16 = None
        self.combined_model = None
        self.modalite_mapping = None
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
        Load text utilities including tokenizer and LSTM model.
        """
        model_path = self.get_model_path()
        with open(os.path.join(model_path,'tokenizer.pkl'), 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer
        self.lstm = load_model(os.path.join(model_path,'image_model.keras'))

    def load_img_utils(self):
        """
        Load image utilities including VGG16 model.
        """
        model_path = self.get_model_path()

        self.vgg16 = load_model(os.path.join(model_path,'image_model.keras'))

    def load_model_utils(self):
        """
        Load model utilities including mapper and combined model.
        """
        model_path = self.get_model_path()
        
        with open(os.path.join(model_path,'modalite_mapping.pkl'), 'rb') as handle:
            modalite_mapping = pickle.load(handle)
        
        self.modalite_mapping = modalite_mapping
        self.inv_modalite_mapping = {value: key for key, value in modalite_mapping.items()}
        
        self.combined_model = load_model(os.path.join(model_path,'combined_model.keras'))

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

    def _predict(self,text_designation, text_description, image):
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

        # Prédiction avec le modèle combiné
        text_pred, image_pred, combined_pred = None, None, None
        
        # Prédiction avec le modèle de texte seul
        text_pred = self.text_model.predict(padded_sequence)

        # Prédiction avec le modèle d'image seul
        image_pred = self.image_model.predict(img_array)

        # Prédiction avec le modèle combiné
        combined_pred = self.combined_model.predict([padded_sequence, img_array])

        # Décoder les prédictions
        text_pred_class = np.argmax(text_pred, axis=1)[0]
        image_pred_class = np.argmax(image_pred, axis=1)[0]
        combined_pred_class = np.argmax(combined_pred, axis=1)[0]

        return {
            "text_prediction": {
                "class": text_pred_class,
                "probability": float(text_pred[0][text_pred_class])
            },
            "image_prediction": {
                "class": image_pred_class,
                "probability": float(image_pred[0][image_pred_class])
            },
            "combined_prediction": {
                "class": combined_pred_class,
                "probability": float(combined_pred[0][combined_pred_class])
            }
        }

    def _predict_from_dataframe(self,df):
        predictions = []

        for index, row in df.iterrows():
            text_designation = row['designation']
            text_description = row['description']
            image_path = row['image_path']

            prediction_result = self.predict_single_data(text_designation, text_description, image_path)
            predictions.append({
                "text_designation": text_designation,
                "text_description": text_description,
                "image_path": image_path,
                "predictions": prediction_result
            })

        return predictions
    
    def predict(self,text_designation, text_description, image):
        result = self._predict(text_designation, text_description, image)
        result = result['combined_prediction']['class']
        result = self.inv_modalite_mapping[result]
        return result

    # def agg_prediction(self, txt_prob, img_prob):
    #     """
    #     Aggregate predictions from text and image models.
        
    #     Args:
    #         txt_prob (np.array): Probability distribution from text model.
    #         img_prob (np.array): Probability distribution from image model.
        
    #     Returns:
    #         int: Aggregated predicted class.
    #     """
    #     concatenate_proba = (
    #         self.best_weights[0] * txt_prob + self.best_weights[1] * img_prob
    #     )
    #     pred = np.argmax(concatenate_proba)
    #     return int(self.mapper[str(pred)])
    
# Utilisation de la classe Model

# model = tf_trimodel(model_type='production', version='latest')
# prediction = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg'))
# print(prediction)

# prediction_from_bytes = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', image_bytes)
# print(prediction_from_bytes)
