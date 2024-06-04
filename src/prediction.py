import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import math
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("french"))

with open("src/models/model_parameters/mapper.json", "r") as json_file:
    mapper = json.load(json_file)

def load_text_model(pathway : str):
    """
    Loading of the trained lstm model located at the pathway
    """
    return keras.models.load_model(pathway)

def preprocess(text):
    """
    Preparation of the text as an argument for prediction
    """
    if isinstance(text, float) and math.isnan(text):
            return ""
        # Supprimer les balises HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Supprimer les caractères non alphabétiques
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Tokenization
    words = word_tokenize(text.lower())
    # Suppression des stopwords et lemmatisation
    filtered_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
        ]
    return " ".join(filtered_words[:10])

def text_predict(entry):
    """
    Text prediction
    Returns: [prdtypecode,sequence of probabilities]
     """
    with open("src/models/model_parameters/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    textes = preprocess(entry)
    sequences=tokenizer.texts_to_sequences([textes])
    padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )
    lstm=load_text_model("models/best_lstm_model.h5")
    lstm_proba=lstm.predict([padded_sequences])

    final_prediction = np.argmax(lstm_proba)
    
    return [mapper[str(final_prediction)],lstm_proba] 

def load_vgg16_model(pathway : str):
    return keras.models.load_model("models/best_vgg16_model.h5")

def image_predict(imageid : str = None, productid : str = None, directory : str =None, nouveau : str = None, picture = None):
    """
    Image prediction following 2 ways :
        _with the directory where the image named following imageid and productid is stored
        _with a complete new pathway, nme included
        _with un object PIL image
    Returns : [prdtypecode,sequence of probabilities]
    """
    imagepath=f"{directory}/image_{imageid}_product_{productid}.jpg"
    if nouveau is not None:
        imagepath=nouveau
    img=None
    if picture is None:
        img = load_img(imagepath, target_size=(224, 224, 3))
    else:
        img=picture
    img_array = img_to_array(img)        
    img_array = preprocess_input(img_array)
    images = tf.convert_to_tensor([img_array], dtype=tf.float32) # vérifier si le passage en liste est correct

    vgg16=keras.models.load_model("models/best_vgg16_model.h5")
    vgg16_proba = vgg16.predict([images]) 
    final_prediction = np.argmax(vgg16_proba)
    return [mapper[str(final_prediction)],vgg16_proba]

def predict(entry : str, imageid : str = None, productid : str = None, directory : str =None, nouveau : str = None):
    """
    Prediction from both designation and image
    Returns: prdtypecode
    """
    best_weights=None
    with open("models/best_weights.pkl","rb") as file :
        best_weights=pickle.load(file)
    #print(best_weights)  #######
    vgg16_proba=image_predict(imageid, productid, directory, nouveau)[1]
    lstm_proba=text_predict(entry)[1]
    #print(vgg16_proba)
    #print(lstm_proba)
    concatenate_proba = (best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba)
    #print(concatenate_proba)
    final_prediction = np.argmax(concatenate_proba)
    #print(final_prediction)
    return mapper[str(final_prediction)]

def main():
    print(text_predict("import japon jeu vidéo toon"))
    print(image_predict(imageid=234234,productid=184251,directory='data/preprocessed/image_train/'))
    print(predict(entry="import japon jeu vidéo toon",
                  imageid=234234,productid=184251,directory='data/preprocessed/image_train/'))

if __name__=="__main__":
    main()