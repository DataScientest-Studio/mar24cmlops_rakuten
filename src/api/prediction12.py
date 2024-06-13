import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import os
from io import BytesIO

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("french"))

with open("src/train_model_legacy/models/model_parameters/mapper.json", "r") as json_file:
    mapper = json.load(json_file)

vgg16=keras.models.load_model("models/production_model/best_vgg16_model.h5")
lstm=keras.models.load_model("models/production_model/best_lstm_model.h5")
best_weights=None
with open("models/production_model/best_weights.pkl","rb") as file :
    best_weights=pickle.load(file)

def new_model(pathway : str = None):  # A tester
    """
    Function to call if the model to work with is not a production model : 
    allows the actualization of the vgg16, lstm and best_weights variables
    (by default, actualizes with the model in production)
    Argument:
        pathway : name of the folder in models/staging where the model resides
    """
    global vgg16, lstm, best_weights  # Non nécessaire si ce sont des objets
    if new_model is not None:
        # Verifies the existence of the model and downloads it else raise an exceptio
        if os.path.isdir(f"models/staging_models/{pathway}"):
            vgg16=keras.models.load_model(f"models/staging_models/{pathway}/best_vgg16_model.h5")
            lstm=keras.models.load_model("models/staging_models/{pathway}/best_lstm_model.h5")
            with open("models/staging_models/{pathway}/best_weights.pkl","rb") as file :
                best_weights=pickle.load(file)
        else:
            raise Exception("Bad location for a model")
    # By default, without any argument, uses the model in production
    else :
        vgg16=keras.models.load_model("models/production_model/best_vgg16_model.h5")
        lstm=keras.models.load_model("models/production_model/best_lstm_model.h5")
        with open("models/best_weights.pkl","rb") as file :
            best_weights=pickle.load(file)

def preprocess(text):
    """
    Preparation of the text as an argument for prediction
    Argument:
        text : the sequence to work with
    Returns:
        the tenth most frequent words as a string
    """
    if isinstance(text, float) and math.isnan(text):
            return ""
    # Removes the HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Removes the non alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Tokenization
    words = word_tokenize(text.lower())
    # REmoves the stop wordds and lemmatization
    filtered_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
        ]
    return " ".join(filtered_words[:10])

def text_predict(entry):
    """
    Text prediction
    Argument:
        entry : the text sequence to evaluate
    Returns:
        [prdtypecode,sequence of probabilities]
     """
    with open("src/train_model_legacy/models/model_parameters/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
    textes = preprocess(entry)
    sequences=tokenizer.texts_to_sequences([textes])
    padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )
    lstm_proba=lstm.predict([padded_sequences])
    final_prediction = np.argmax(lstm_proba)   
    return [mapper[str(final_prediction)],lstm_proba] 

def image_predict(imageid : int = None, productid : int = None, directory : str =None, new_image : str = None):
    """
    Image prediction following 2 ways :
        _with the directory where the image named following imageid and productid is stored
        _with a complete new pathway, name included
        _with un object PIL image
    Returns : [prdtypecode,sequence of probabilities]
    """
    imagepath=None
    #imageid=str(imageid)
    #productid=str(productid)
    #imagepath=f"{directory}/image_{imageid}_product_{productid}.jpg"
    if imageid is not None and productid is not None and directory is not None:
        imagepath=os.path.join(directory,f"image_{imageid}_product_{productid}.jpg")
    # if specified, new_image has the priority over imageid, productid and directory
    if new_image is not None:
        imagepath=new_image
    img=load_img(imagepath,target_size=(224,224,3))
    img_array = img_to_array(img)        
    img_array = preprocess_input(img_array)
    images = tf.convert_to_tensor([img_array], dtype=tf.float32)
    vgg16_proba = vgg16.predict([images]) 
    final_prediction = np.argmax(vgg16_proba)
    return [mapper[str(final_prediction)],vgg16_proba]

def image_predict_object(picture):
    """
    Image prediction following 2 ways :
        _with the directory where the image 
        _with a complete new pathway, nme included
        _with un object PIL image
    Returns : [prdtypecode,sequence of probabilities]
    """
    img=picture
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    images = tf.convert_to_tensor([img_array], dtype=tf.float32)
    vgg16_proba = vgg16.predict([images]) 
    final_prediction = np.argmax(vgg16_proba)
    return [mapper[str(final_prediction)],vgg16_proba]

# Fonction qui sera inutile dans la version finale
def predict(entry : str, imageid : int = None, productid : int = None, directory : str =None, new_image : str = None):
    """
    Prediction from both designation and image
    Returns: prdtypecode
    """
    vgg16_proba=image_predict(imageid, productid, directory, new_image)[1]
    lstm_proba=text_predict(entry)[1]
    concatenate_proba = (best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba)
    final_prediction = np.argmax(concatenate_proba)
    return mapper[str(final_prediction)]

def predictionT(designation : str =None, imageid : int = None, productid : int = None, directory : str = 'data/preprocessed/image_train', new_image : str = None, file = None):
    """
    Function that takes all the possible entries and returns a prdtypecode
    Its aim is to unify all the possible entries configurations into one unique function to simplify the use
    and the creation of the api.
    Arguments: 
        designation : (optional) the designation of the product
        imageid : (optional) the imageid associated to the product if already recorded
        productid : (optional) the productid associated to the product if already recorded
        directory : (optional) the folder where the associated image is stored
        new_image : (optional) the complete name of an new image, complete pathway included
        file : (optional) the PIL (tensorflow version) object representing the image
    Returns:
        the prdtypecode if success
        an exception if failed
    """
    lstm_return=None
    vgg16_return=None
    img=None
    if designation is not None :
        lstm_return=text_predict(designation)
    if imageid is not None and productid is not None:
        if os.path.isfile(os.path.join(directory,f"image_{imageid}_product_{productid}.jpg")) is False:
            raise Exception("Mauvaises références pour retrouver l\'image")
        vgg16_return=image_predict(imageid=imageid,productid=productid,directory=directory)
    elif new_image is not None:
        vgg16_return=image_predict(new_image=new_image)
    if file is not None:
        img=load_img(BytesIO(file),target_size=(224,224,3))
        vgg16_return=image_predict_object(img)
    if lstm_return is not None and vgg16_return is None:
        return lstm_return[0]
    if vgg16_return is not None and lstm_return is None:
        return vgg16_return[0]
    if lstm_return is not None and vgg16 is not None:
        concatenate_proba = (best_weights[0] * lstm_return[1] + best_weights[1] * vgg16_return[1])
        final_prediction = np.argmax(concatenate_proba)
        return mapper[str(final_prediction)]
    raise Exception("La prédiction n\'a pas pu aboutir.")


def main():
    pass

if __name__=="__main__":
    main()
