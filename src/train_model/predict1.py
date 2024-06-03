from features.build_features import TextPreprocessor
from features.build_features import ImagePreprocessor
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd
import argparse



class Predict:
    def __init__(
        self,
        tokenizer,
        lstm,
        vgg16,
        best_weights,
        mapper,
        filepath,
        imagepath,
        df
    ):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper
        self.filepath = filepath
        self.imagepath = imagepath,
        self.df=df
        # Charger les configurations et modèles
        if self.tokenizer is None:
            with open("models/model_parameters/tokenizer_config.json", "r", encoding="utf-8") as json_file:
                tokenizer_config = json_file.read()
            self.tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
        if self.lstm is None:
            self.lstm = keras.models.load_model("models/best_lstm_model.h5")
        if self.vgg16 is None:
            self.vgg16 = keras.models.load_model("models/best_vgg16_model.h5")
        if self.best_weights is None:
            with open("models/best_weights.json", "r") as json_file:
                self.best_weights = json.load(json_file)
        if self.mapper is None :
            with open("models/mapper.json", "r") as json_file:
                self.mapper = json.load(json_file)
        # Curieux : beaucoup de ces éléments sont dans main alors qu'ils sont systématiquement utilisés
        # par Predict : pourquoi n'ont-ils pas tout mis ici dès le départ au lieu de les mettre 
        # en paramètres ?
        
    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self):
        X=None
        if self.df is None :
            X = pd.read_csv(self.filepath)[:10] ############### Lever la limitation au besoin
        else:
            X=self.df
        
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor(self.imagepath)
        text_preprocessor.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)

        sequences = self.tokenizer.texts_to_sequences(X["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )

        target_size = (224, 224, 3)
        images = X["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        lstm_proba = self.lstm.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([images])

        concatenate_proba = (
            self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)

        return {
            i: self.mapper[str(final_predictions[i])]
            for i in range(len(final_predictions))
        }



def main():
    parser = argparse.ArgumentParser(description= "Input data")
    
    parser.add_argument("--dataset_path", default = "data/preprocessed/X_train_update.csv", type=str,help="File path for the input CSV file.")
    parser.add_argument("--images_path", default = "data/preprocessed/image_train", type=str,  help="Base path for the images.")
    args = parser.parse_args()

    # Charger les configurations et modèles
    with open("models/model_parameters/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    lstm = keras.models.load_model("models/best_lstm_model.h5")
    vgg16 = keras.models.load_model("models/best_vgg16_model.h5")

    with open("models/best_weights.json", "r") as json_file:
        best_weights = json.load(json_file)

    with open("models/model_parameters/mapper.json", "r") as json_file:
        mapper = json.load(json_file)
        
    
    predictor = Predict(
        tokenizer=tokenizer,
        lstm=lstm,
        vgg16=vgg16,
        best_weights=best_weights,
        mapper=mapper,
        filepath= args.dataset_path,
        imagepath = args.images_path,
    )

    # Création de l'instance Predict et exécution de la prédiction
    predictions = predictor.predict()

    # Sauvegarde des prédictions
    with open("data/preprocessed/predictions.json", "w", encoding="utf-8") as json_file:
        json.dump(predictions, json_file, indent=2)


if __name__ == "__main__":
    main()