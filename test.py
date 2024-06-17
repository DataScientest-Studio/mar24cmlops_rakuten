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

vgg16=keras.models.load_model("models/production_model/best_vgg16_model.h5")

with open("src/train_model_legacy/models/model_parameters/mapper.json", "r") as json_file:
    mapper = json.load(json_file)
    
def image_predict(imagepath):
    """
    Image prediction following 2 ways :
        _with the directory where the image named following imageid and productid is stored
        _with a complete new pathway, name included
        _with un object PIL image
    Returns : [prdtypecode,sequence of probabilities]
    """

    img=load_img(imagepath,target_size=(224,224,3))
    img_array = img_to_array(img)        
    img_array = preprocess_input(img_array)
    images = tf.convert_to_tensor([img_array], dtype=tf.float32)
    vgg16_proba = vgg16.predict([images]) 
    final_prediction = np.argmax(vgg16_proba)
    return [mapper[str(final_prediction)],vgg16_proba]

if __name__ == '__main__':
    image_path = 'data/12.jpg'
    pred = image_predict(imagepath=image_path)
    print(pred)
