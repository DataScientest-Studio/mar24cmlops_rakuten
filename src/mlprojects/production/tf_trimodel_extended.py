import os
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.tensorflow
from api.utils.resolve_path import resolve_path
from mlprojects.production.tf_trimodel import tf_trimodel

class tf_trimodel_extended(tf_trimodel):
    """
    Extended class of tf_trimodel with additional methods to retrain the model
    and the aggregation layer.
    """

    def __init__(self, model_name, version, model_type):
        """
        Initialize the tf_trimodel_extended class.

        Args:
            model_name (str): Name of the model.
            version (str): Version of the model.
            model_type (str): Type of the model (production or staging).
        """
        super().__init__(model_name, version, model_type)
        self.retrained_base_path = resolve_path(
            (f"models/staging_models/{model_name}/")
        )
        self.is_retrained = False
        self.num_epochs = 15
        self.num_listings = 5000
        self.batch_size = 32

    def load_and_preprocess_image(self, path):
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        return img_array

    def process_df_for_train(self, df):
        texts = (df["description"] + " " + df["designation"]).astype(str)
        df["image_path"] = df.apply(
            lambda row: resolve_path(
                f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
            ),
            axis=1,
        )
        image_paths = df["image_path"].values
        images = np.array(
            [self.load_and_preprocess_image(path) for path in image_paths]
        )
        return texts, images

    def process_texts_for_train(self, texts):
        sequences = self.new_tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=100)
        return padded_sequences

    def make_modalite_mapping_and_labels(self, df):
        modalite_mapping = {
            int(modalite): i for i, modalite in enumerate(df["prdtypecode"].unique())
        }
        df["target_prdtypecode"] = df["prdtypecode"].replace(modalite_mapping)
        labels = df["target_prdtypecode"]
        num_classes = len(np.unique(labels))
        labels = to_categorical(labels, num_classes=num_classes)

        with open(
            os.path.join(self.retrained_base_path, "modalite_mapping.pkl"), "wb"
        ) as pickle_file:
            pickle.dump(modalite_mapping, pickle_file)

        return modalite_mapping, labels, num_classes

    def create_text_model(self, num_classes):
        text_input = tf.keras.Input(shape=(100,), name="text_input")
        x = tf.keras.layers.Embedding(input_dim=10000, output_dim=64)(text_input)
        x = tf.keras.layers.LSTM(128)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        text_output = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="text_output"
        )(x)
        return tf.keras.Model(inputs=text_input, outputs=text_output)

    def create_image_model(self, num_classes):
        image_input = tf.keras.Input(shape=(224, 224, 3), name="image_input")
        base_model = tf.keras.applications.VGG16(
            include_top=False, input_tensor=image_input
        )
        base_model.trainable = False  # Fine-tuning
        x = tf.keras.layers.Flatten()(base_model.output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        image_output = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="image_output"
        )(x)
        return tf.keras.Model(inputs=image_input, outputs=image_output)

    def create_combined_model(self, num_classes, text_model, image_model):
        combined_input = tf.keras.layers.concatenate(
            [text_model.output, image_model.output]
        )
        x = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="combined_output"
        )(combined_input)
        return tf.keras.Model(inputs=[text_model.input, image_model.input], outputs=x)

    def train_model(self, new_df):
        """
        Retrain the models with new data.

        Args:
            new_df (DataFrame): New data for retraining.
            epochs (int): Number of epochs for retraining.
            batch_size (int): Batch size for retraining.
        """
        print(self.retrained_base_path)
        self.is_retrained = True
        self.retrained_version = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        self.retrained_base_path = os.path.join(
            self.retrained_base_path, self.retrained_version
        )

        # Create the directory if it does not exist
        os.makedirs(self.retrained_base_path, exist_ok=True)

        print(self.retrained_base_path)
        # Update tokenizer with new text data
        self.new_tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.new_tokenizer.fit_on_texts(
            (new_df["description"] + " " + new_df["designation"]).astype(str)
        )

        # Sauvegarde du tokenizer
        with open(
            os.path.join(self.retrained_base_path, "tokenizer.pkl"), "wb"
        ) as handle:
            pickle.dump(self.new_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        texts, images = self.process_df_for_train(new_df)
        padded_sequences = self.process_texts_for_train(texts)

        modalite_mapping, labels, num_classes = self.make_modalite_mapping_and_labels(
            new_df
        )

        X_train_texts, X_test_texts, X_train_images, X_test_images, y_train, y_test = (
            train_test_split(
                padded_sequences, images, labels, test_size=0.2, random_state=42
            )
        )

        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        try:
            mlflow.set_experiment(self.model_name)
            mlflow_active = True
        except Exception as e:
            print(f"MLflow experiment setup failed: {e}")
            mlflow_active = False

        try:
            if mlflow_active:
                mlflow.start_run()
            if mlflow_active:
                mlflow.log_param("num_epochs", self.num_epochs)
                mlflow.log_param("batch_size", self.batch_size)
                
            text_model = self.create_text_model(num_classes)
            text_model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            history_text = text_model.fit(
                X_train_texts,
                y_train,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                validation_data=(X_test_texts, y_test),
            )

            if mlflow_active:
                # Log text model metrics
                mlflow.log_metrics({
                    "text_train_accuracy": history_text.history['accuracy'][-1],
                    "text_val_accuracy": history_text.history['val_accuracy'][-1],
                    "text_train_loss": history_text.history['loss'][-1],
                    "text_val_loss": history_text.history['val_loss'][-1]
                })

                # Sauvegarder le modèle textuel
                text_model_path = os.path.join(self.retrained_base_path, "text_model.keras")
                text_model.save(text_model_path)
                mlflow.keras.log_model(text_model, "text_model")

            image_model = self.create_image_model(num_classes)
            image_model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            history_image = image_model.fit(
                X_train_images,
                y_train,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                validation_data=(X_test_images, y_test),
            )

            if mlflow_active:
                # Log image model metrics
                mlflow.log_metrics({
                    "image_train_accuracy": history_image.history['accuracy'][-1],
                    "image_val_accuracy": history_image.history['val_accuracy'][-1],
                    "image_train_loss": history_image.history['loss'][-1],
                    "image_val_loss": history_image.history['val_loss'][-1]
                })

                # Sauvegarder le modèle d'image
                image_model_path = os.path.join(self.retrained_base_path, "image_model.keras")
                image_model.save(image_model_path)
                mlflow.keras.log_model(image_model, "image_model")

            # Désactiver l'entraînement pour les modèles préentraînés
            text_model.trainable = False
            image_model.trainable = False

            combined_model = self.create_combined_model(
                num_classes, text_model, image_model
            )
            combined_model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            history_combined = combined_model.fit(
                [X_train_texts, X_train_images],
                y_train,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                validation_data=([X_test_texts, X_test_images], y_test),
            )

            if mlflow_active:
                # Log combined model metrics
                mlflow.log_metrics({
                    "combined_train_accuracy": history_combined.history['accuracy'][-1],
                    "combined_val_accuracy": history_combined.history['val_accuracy'][-1],
                    "combined_train_loss": history_combined.history['loss'][-1],
                    "combined_val_loss": history_combined.history['val_loss'][-1]
                })

                # Sauvegarder le modèle combiné
                combined_model_path = os.path.join(self.retrained_base_path, "combined_model.keras")
                combined_model.save(combined_model_path)
                mlflow.keras.log_model(combined_model, "combined_model")

        except Exception as e:
            print(f"MLflow logging failed: {e}")
        finally:
            if mlflow_active:
                mlflow.end_run()

# Exemple d'utilisation
import pandas as pd
X_train = pd.read_csv(resolve_path("data/X_train.csv"), index_col=0)
Y_train = pd.read_csv(resolve_path("data/Y_train.csv"), index_col=0)
listing_df = X_train.join(Y_train)

listing_df = listing_df.sample(5000)

# Initialiser l'extension du modèle
extended_model = tf_trimodel_extended("tf_trimodel", "20240708_19-15-54", "production")

# Réentraîner le modèle avec les nouvelles données
extended_model.train_model(listing_df)

# Autre exemple de batch predict
# import pandas as pd

# X_train = pd.read_csv(resolve_path("data/X_train.csv"), index_col=0)
# Y_train = pd.read_csv(resolve_path("data/Y_train.csv"), index_col=0)
# listing_df = X_train.join(Y_train)

# listing_df = listing_df.sample(10)
# listing_df["image_path"] = listing_df.apply(
#     lambda row: resolve_path(
#         f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
#     ),
#     axis=1,
# )
# model = tf_trimodel_extended("tf_trimodel", "20240714_22-20-29", "staging")

# result = model._predict_from_dataframe(listing_df)

# print(pd.DataFrame(result))
