import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils.resolve_path import resolve_path
import duckdb
import pickle
import json
import os
import re
from dotenv import load_dotenv
from datetime import datetime
from mlflow import MlflowClient
import mlflow

# Load environment variables from .env file
load_dotenv(resolve_path(".env/.env.development"))

aws_config_path = resolve_path(os.environ["AWS_CONFIG_PATH"])
duckdb_path = os.path.join(
    resolve_path(os.environ["DATA_PATH"]), os.environ["RAKUTEN_DB_NAME"].lstrip("/")
)
rakuten_db_name = os.environ["RAKUTEN_DB_NAME"]


class production_model_retrain:
    def __init__(
        self, model_type="production", version=None, model_name=None, is_production=True
    ):
        self.model_type = model_type
        if version is not None:
            self.version = version
        else:
            self.version = datetime.now().strftime("%Y%m%d_%H-%M-%S")
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
        self.is_production = is_production
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.load_txt_utils()
        self.load_model_utils()
        self.load_duckdb_utils()
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        self.data_path = self.get_model_path()
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        print(self.data_path)  ##########
        print(
            resolve_path(os.path.join(self.data_path, "best_lstm_model.keras"))
        )  ##########

        # Define tracking_uri
        self.client = MlflowClient(
            tracking_uri="http://127.0.0.1:8080"
        )  ## Attention à l'adresse
        # Define experiment name, run name and artifact_path name
        self.apple_experiment = mlflow.set_experiment(__name__)
        self.run_name = f"{self.model_name}/{self.version}/"
        self.artifact_path = "retrain"
        self.metrics = {}

    def get_model_path(self):
        folder = "staging_models"
        base_path = resolve_path(f"models/{folder}/{self.model_name}/{self.version}/")
        return base_path

    def load_txt_utils(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))

    def load_model_utils(self):
        model_path = "models"

        with open(
            resolve_path(os.path.join(model_path, "mapper.json")), "r"
        ) as json_file:
            self.mapper = json.load(json_file)

    def load_duckdb_utils(self):
        self.db_conn = duckdb.connect(database=duckdb_path, read_only=False)

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

    def process_img(selg, img):
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

    def split_train_test(
        self,
        df,
        train_ratio=5 / 7.0,
        test_ratio=1 / 7.0,
        test_val=1 / 7.0,
        limit_train=27 * 600,
        limit_test=27 * 50,
        limit_val=27 * 50,
    ):
        """
        Split the DataFrame following the various provided ratios and truncated if necessary to limit the following train time
        Args:
            df (_type_): _description_
            train_ratio (_type_, optional): _description_. Defaults to 5/7..
            test_ratio (_type_, optional): _description_. Defaults to 1/7..
            test_val (_type_, optional): _description_. Defaults to 1/7..
            limit_train (_type_, optional): _description_. Defaults to 27*600.
            limit_test (_type_, optional): _description_. Defaults to 27*50.
            limit_val (_type_, optional): _description_. Defaults to 27*50.

        Returns:
            X_train,X_val,X_test,y_train,y_val,y_test
        """
        X = df.drop(["prdtypecode"], axis=1)
        y = df["prdtypecode"]
        X_train1_init, X_test1, y_train1_init, y_test1 = train_test_split(
            X, y, test_size=test_ratio, random_state=42, stratify=y
        )
        X_train1, X_val1, y_train1, y_val1 = train_test_split(
            X_train1_init,
            y_train1_init,
            test_size=test_ratio / (train_ratio + test_ratio),
            random_state=42,
            stratify=y_train1_init,
        )
        X_train, y_train = RandomOverSampler().fit_resample(X_train1, y_train1)
        X_test, y_test = RandomOverSampler().fit_resample(X_test1, y_test1)
        X_val, y_val = RandomOverSampler().fit_resample(X_val1, y_val1)
        return (
            X_train[:limit_train],
            X_val[:limit_val],
            X_test[:limit_test],
            y_train[:limit_train],
            y_val[:limit_val],
            y_test[:limit_test],
        )

    def data_handle(self, db_limitation=False, init=False):
        self.db_limitation = db_limitation
        self.init = init
        liste_valeur = None
        if init is False:
            if db_limitation is True:
                liste_valeur = self.db_conn.sql(
                    "SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user IS NOT NULL AND NOT user='init_user' AND user_prdtypecode IS NOT NULL;"
                ).df()
            else:
                liste_valeur = self.db_conn.sql(
                    "SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user IS NOT NULL AND user_prdtypecode IS NOT NULL;"
                ).df()
        else:
            liste_valeur = self.db_conn.sql(
                "SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user='init_user';"
            ).df()
        mapper_inv = {self.mapper[i]: i for i in self.mapper.keys()}
        liste_valeur["prdtypecode"] = (
            liste_valeur["prdtypecode"].astype(str).replace(mapper_inv)
        )
        liste_valeur["description"] = liste_valeur["designation"] + str(
            liste_valeur["description"]
        )
        liste_valeur.drop(["designation"], axis=1, inplace=True)
        filepath = resolve_path(
            "data/preprocessed/image_train"
        )  # limitation à un seul dossier. resolve_path ?
        liste_valeur["image_path"] = liste_valeur.apply(
            lambda row: resolve_path(
                f"{filepath}/image_{row['imageid']}_product_{row['productid']}.jpg"
            ),
            axis=1,
        )
        liste_valeur.dropna(
            axis=0, subset=["image_path", "description"], how="any", inplace=True
        )
        liste_valeur["description"] = liste_valeur["description"].apply(
            self.preprocess_txt
        )
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = (
            self.split_train_test(liste_valeur)
        )

    def preprocess_txt(self, text):
        if text is not None:
            text = re.sub(r"[^a-zA-Z]", " ", text)
            words = word_tokenize(text.lower())
            filtered_words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words and word != "none"
            ]
            return " ".join(filtered_words[:10])
        return None

    def tokenize(self):
        print(self.X_train["description"][:5])
        self.tokenizer.fit_on_texts(self.X_train["description"])
        tokenizer_config = self.tokenizer.to_json()
        with open(os.path.join(self.data_path, "tokenizer_config.json"), "w") as file:
            file.write(tokenizer_config)

    def text_train(self, epochs=1):  #############" paramètres ?"
        self.tokenize()
        print(self.X_train["description"][:10])  ##############
        train_sequences = self.tokenizer.texts_to_sequences(self.X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences, maxlen=10, padding="post", truncating="post"
        )
        val_sequences = self.tokenizer.texts_to_sequences(self.X_val["description"])
        val_padded_sequences = pad_sequences(
            val_sequences, maxlen=10, padding="post", truncating="post"
        )

        text_input = Input(shape=(10,))
        embedding_layer = Embedding(input_dim=10000, output_dim=128)(text_input)
        lstm_layer = LSTM(128)(embedding_layer)
        output = Dense(27, activation="softmax")(lstm_layer)

        self.lstm = Model(inputs=[text_input], outputs=output)

        if self.is_production is True:
            pathway = "models/production_model"
            self.lstm = keras.models.load_model(
                resolve_path(os.path.join(pathway, "best_lstm_model.keras"))
            )

        self.lstm.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        lstm_callbacks = [
            ModelCheckpoint(
                filepath=resolve_path(
                    os.path.join(self.data_path, "best_lstm_model.keras")
                ),
                save_best_only=True,
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.lstm_history = self.lstm.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(self.y_train, num_classes=27),
            epochs=epochs,
            batch_size=32,
            validation_data=(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(self.y_val, num_classes=27),
            ),
            callbacks=lstm_callbacks,
        )
        self.metrics["lstm_acc"] = max(self.lstm_history.history["accuracy"])
        self.metrics["lstm_val_acc"] = max(self.lstm_history.history["val_accuracy"])
        self.metrics["lstm_loss"] = min(self.lstm_history.history["loss"])
        self.metrics["lstm_val_loss"] = min(self.lstm_history.history["val_loss"])

    def image_train(self, epochs=1):  ######### Arguments
        batch_size = 32
        num_classes = 27

        df_train = pd.concat([self.X_train, self.y_train.astype(str)], axis=1)
        df_val = pd.concat([self.X_val, self.y_val.astype(str)], axis=1)

        # Créer un générateur d'images pour le set d'entraînement
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )  # Normalisation des valeurs de pixel
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),  # Adapter à la taille d'entrée de VGG16
            batch_size=batch_size,
            class_mode="categorical",  # Utilisez 'categorical' pour les entiers encodés en one-hot
            shuffle=True,
        )

        # Créer un générateur d'images pour le set de validation
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )  # Normalisation des valeurs de pixel
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,  # Pas de mélange pour le set de validation
        )

        image_input = Input(
            shape=(224, 224, 3)
        )  # Adjust input shape according to your images

        vgg16_base = VGG16(
            include_top=False, weights="imagenet", input_tensor=image_input
        )

        x = vgg16_base.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)  # Add some additional layers if needed
        output = Dense(num_classes, activation="softmax")(x)

        self.vgg16 = Model(inputs=vgg16_base.input, outputs=output)

        for layer in vgg16_base.layers:
            layer.trainable = False

        if self.is_production is True:
            pathway = "models/production_model/"
            self.vgg16 = keras.models.load_model(
                resolve_path(os.path.join(pathway, "best_vgg16_model.keras"))
            )

        self.vgg16.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        vgg_callbacks = [
            ModelCheckpoint(
                filepath=resolve_path(
                    os.path.join(self.data_path, "best_vgg16_model.keras")
                ),
                save_best_only=True,
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.vgg16_history = self.vgg16.fit(
            train_generator,
            epochs=epochs,  ############# le temps de l'évaluation
            validation_data=val_generator,
            callbacks=vgg_callbacks,
        )
        self.metrics["vgg16_acc"] = max(self.vgg16_history.history["accuracy"])
        self.metrics["vgg16_val_acc"] = max(self.vgg16_history.history["val_accuracy"])
        self.metrics["vgg16_loss"] = min(self.vgg16_history.history["loss"])
        self.metrics["vgg16_val_loss"] = min(self.vgg16_history.history["val_loss"])

    def concatenate_train(self, limitation_per_class=50):
        new_X_train = pd.DataFrame(columns=self.X_train.columns)
        new_y_train = pd.DataFrame(
            columns=[0]
        )  # Créez la structure pour les étiquettes

        # Boucle à travers chaque classe
        for class_label in range(27):
            # Indices des échantillons appartenant à la classe actuelle
            indices = np.where(self.y_train == str(class_label))[0]

            # Sous-échantillonnage aléatoire pour sélectionner 'new_samples_per_class' échantillons
            sampled_indices = resample(
                indices, n_samples=limitation_per_class, replace=False, random_state=42
            )

            # Ajout des échantillons sous-échantillonnés et de leurs étiquettes aux DataFrames
            new_X_train = pd.concat([new_X_train, self.X_train.loc[sampled_indices]])
            new_y_train = pd.concat([new_y_train, self.y_train.loc[sampled_indices]])

        # Réinitialiser les index des DataFrames
        new_X_train = new_X_train.reset_index(drop=True)
        new_y_train = new_y_train.reset_index(drop=True)[["prdtypecode"]]
        new_y_train = new_y_train.values.reshape(1350).astype("int")

        train_sequences = self.tokenizer.texts_to_sequences(new_X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences, maxlen=10, padding="post", truncating="post"
        )

        # Paramètres pour le prétraitement des images

        images_train = new_X_train["image_path"].apply(
            lambda x: self.process_img(self.path_to_img(x))
        )

        images_train = tf.convert_to_tensor(images_train.tolist(), dtype=tf.float32)

        lstm_proba = self.lstm.predict([train_padded_sequences])

        vgg16_proba = self.vgg16.predict([images_train])

        # Recherche des poids optimaux en utilisant la validation croisée
        best_weights = None
        best_accuracy = 0.0

        for lstm_weight in np.linspace(0, 1, 101):  # Essayer différents poids pour LSTM
            vgg16_weight = 1.0 - lstm_weight  # Le poids total doit être égal à 1

            combined_predictions = (lstm_weight * lstm_proba) + (
                vgg16_weight * vgg16_proba
            )
            final_predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(new_y_train, final_predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (lstm_weight, vgg16_weight)

        self.metrics["agg_acc"] = best_accuracy

        with open(
            resolve_path(os.path.join(self.data_path, "best_weights.pkl")), "wb"
        ) as file:
            pickle.dump(best_weights, file)

    def train(self, epochs=1):
        self.text_train(epochs=epochs)
        self.image_train()
        self.concatenate_train()
        # sauvegarde models
        # enregistrement sauvegarde scores

        # Train model
        params = {
            "epochs": epochs,
            "is_production": self.is_production,
            "db_limitation": self.db_limitation,
            "init": self.init,
        }  # Etendre à tous les paramètres d'entrée ?

        # Store information in tracking server
        with mlflow.start_run(run_name=self.run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(self.metrics)
            # mlflow.sklearn.log_model(
            #    sk_model=rf, input_example=X_val, artifact_path=self.artifact_path
            # )
