import os
import json
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from api.utils.resolve_path import resolve_path
from io import BytesIO

# Chargement des données

base_path = resolve_path('data/test_model/')

X_train = pd.read_csv(resolve_path("data/X_train.csv"), index_col=0)
Y_train = pd.read_csv(resolve_path("data/Y_train.csv"), index_col=0)
listing_df = X_train.join(Y_train)

# Prétraitement des textes

listing_df["image_path"] = listing_df.apply(
    lambda row: resolve_path(f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"),
    axis=1,
)

# Charger les modèles sauvegardés
def load_models():
    text_model = load_model(os.path.join(base_path,'text_model.keras'))
    image_model = load_model(os.path.join(base_path,'image_model.keras'))
    combined_model = load_model(os.path.join(base_path,'combined_model.keras'))
    return text_model, image_model, combined_model

# Charger les modèles
text_model, image_model, combined_model = load_models()

# Charger le tokenizer depuis le fichier tokenizer.pkl
with open(os.path.join(base_path,'tokenizer.pkl'), 'rb') as handle:
    tokenizer = pickle.load(handle)
    
def path_to_img(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(np.expand_dims(img_array, axis=0))
    return img_array

def byte_to_img(image_bytes):
    img = load_img(BytesIO(image_bytes), target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(np.expand_dims(img_array, axis=0))
    return img_array
    
def predict_single_data(text_designation, text_description, image):
    # Prétraitement du texte
    text = str(text_description) + " " + str(text_designation)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Chargement et prétraitement de l'image
    if isinstance(image, str):  # Si l'image est un chemin
        img_array = path_to_img(image)
    elif isinstance(image, bytes):  # Si l'image est en bytes
        img_array = byte_to_img(image)
    else:
        raise ValueError("Image must be a file path or bytes.")

    # Prédiction avec le modèle combiné
    text_pred, image_pred, combined_pred = None, None, None
    
    # Prédiction avec le modèle de texte seul
    text_pred = text_model.predict(padded_sequence)

    # Prédiction avec le modèle d'image seul
    image_pred = image_model.predict(img_array)

    # Prédiction avec le modèle combiné
    combined_pred = combined_model.predict([padded_sequence, img_array])

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

def predict_from_dataframe(df):
    predictions = []

    for index, row in df.iterrows():
        text_designation = row['designation']
        text_description = row['description']
        image_path = row['image_path']

        prediction_result = predict_single_data(text_designation, text_description, image_path)
        predictions.append({
            "text_designation": text_designation,
            "text_description": text_description,
            "image_path": image_path,
            "predictions": prediction_result
        })

    return predictions

# Exemple d'utilisation
text_designation = "Jeu video"
text_description = "Titi et les bijoux magiques jeux video enfants gameboy advance"
image_path = resolve_path("data/images/image_train/image_528113_product_923222.jpg")
result = predict_single_data(text_designation, text_description, image_path)
#print(json.dumps(result['predictions'], indent=4))
print(result['combined_prediction']['class'])

# Exemple d'utilisation avec un DataFrame
df = listing_df.iloc[0:2]
results = predict_from_dataframe(df)
print(pd.DataFrame(results))