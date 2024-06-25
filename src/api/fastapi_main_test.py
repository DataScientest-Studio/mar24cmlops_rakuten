from fastapi import FastAPI, HTTPException, Depends, status
#from prediction import predict_with_unified_interface
from fastapi import UploadFile, File
import os
import uvicorn
from dotenv import load_dotenv

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
from io import BytesIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Loads every context to allow text_analysis
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("french"))


#vgg16=keras.models.load_model(os.path.join(prefix,"models/production_model/best_vgg16_model.h5"))
#lstm=keras.models.load_model(os.path.join(prefix,"models/production_model/best_lstm_model.h5"))
#best_weights=None

# Loads the weights for the concatenation of the results of the previous models
#with open(os.path.join(prefix,"models/production_model/best_weights.pkl"),"rb") as file :
#    best_weights=pickle.load(file)






# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def get_index():
    return {'data': 'hello world '+os.getcwd()}

@app.post('/predict')
async def prediction(designation : str =None, imageid : int = None, productid : int = None, directory : str = 'data/preprocessed/image_train', new_image : str = None, file : UploadFile | None = None):
    img_context=None
    if file is not None :
        img_context= await file.read()
    #return {'prediction' : predict_with_unified_interface(designation=designation, imageid=imageid,productid=productid,directory=directory,new_image=new_image,file=img_context)}
    return {'prediction':os.path.join(prefix,"src/train_model_legacy/models/model_parameters/mapper.json")}


if __name__ == "__main__":
    load_dotenv('.env/.env.development')
    # Modification of the pathways following the environnment
    prefix=None
    if os.getenv('DATA_PATH') is None:
        prefix='.'
    else :
        prefix='/app'
    # Loads the mapper between the return of the models and the associated prdtypecode
    with open(os.path.join(prefix,"src/train_model_legacy/models/model_parameters/mapper.json"), "r") as json_file:
        mapper = json.load(json_file)
    lstm=keras.models.load_model(os.path.join(prefix,"models/production_model/best_lstm_model.keras"))
    vgg16=keras.models.load_model(os.path.join(prefix,"models/production_model/best_vgg16_model.keras"))
    uvicorn.run(app, host="0.0.0.0", port=8001)
