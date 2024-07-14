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
    def __init__(self, model_name=None, model_type="production", version="latest"):
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
        #folder = (
        #    "production_model" if self.model_type == "production" else "staging_models"
        #)##""

        # Determine version folder
        #if self.version == "latest":
        #    versions = sorted(
        #        os.listdir(resolve_path(f"models/{folder}/{self.model_name}/"))
        #    )
        #    version = versions[-1]
        #else:
        #    version = self.version

        # Construct base path
        #base_path = resolve_path(f"models/{folder}/{self.model_name}/{version}/")
        base_path=resolve_path(f"models/staging_models/production_model_retrain/20240711_11-09-23/")
        
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
            #sequences = self.tokenizer.texts_to_sequences([textes])
            #padded_sequences = pad_sequences(
            #    sequences, maxlen=10, padding="post", truncating="post"
            #)
            #return padded_sequences
            
            return textes
        return None

    def path_to_img(self, img_path):
        img = load_img(img_path, target_size=(224, 224, 3))
        return img

    def process_img(self, img):
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array


    def agg_prediction(self, txt_prob, img_prob):
        txt_prob=[np.array(x) for x in txt_prob]
        concatenate_proba = (
            self.best_weights[0] * np.array(txt_prob) + self.best_weights[1] * np.array(img_prob)
        )
        pred = np.argmax(concatenate_proba,axis=1)
        sortie = [int(self.mapper[str(x)]) for x in pred]
        return sortie


    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)        
        img_array = preprocess_input(img_array)
        return img_array

    def batch_predict(self,df):
        #df=df[:100] ##### Dans un premier temps
        df.loc[:,['description']]=df['description'].astype(str)+df['designation'].astype(str)
        df['description']=df['description'].apply(self.process_txt)
        sequences = self.tokenizer.texts_to_sequences(df['description'])
        padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")
        txt_prob = self.lstm.predict([padded_sequences])
        vgg_result=[]
        for i in range(0,len(df.index),100):
            images=df[i:i+100]['image'].apply(lambda x : self.path_to_img(x))
            img_array=images.apply(lambda x : self.process_img(x))
            f_imgs = tf.convert_to_tensor(img_array.tolist(),dtype=tf.float32)
            img_prob = self.vgg16.predict([f_imgs])
            vgg_result.extend(list(img_prob))
    
        agg_pred = self.agg_prediction(txt_prob, vgg_result)
        return agg_pred
    

        

# Utilisation de la classe Model

# model = tf_trimodel(model_type='production', version='latest')
# prediction = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg'))
# print(prediction)

# prediction_from_bytes = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', image_bytes)
# print(prediction_from_bytes)