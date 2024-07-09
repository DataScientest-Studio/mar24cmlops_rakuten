import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import os
from io import BytesIO
from dotenv import load_dotenv
import duckdb
from utils.make_db import download_initial_db
from utils.security import create_access_token, verify_password
from utils.s3_utils import create_s3_conn_from_creds, download_from_s3
import boto3 
import pandas as pd 
from prediction import image_predict
from utils.resolve_path import resolve_path
from sklearn.metrics import precision_score

# Load environment variables from .env file
load_dotenv('.env/.env.development')
        
aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
duckdb_path = os.path.join(resolve_path(os.environ['DATA_PATH']), os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']

# Loads the mapper between the return of the models and the associated prdtypecode
with open(resolve_path("models/mapper.json"), "r") as json_file:
    mapper = json.load(json_file)

def text_predict(X,model,date):
    """
    Text prediction
    Argument:
        X : the X_test file already pretreated
    Returns:
        [prdtypecode,sequence of probabilities]
     """

    with open(resolve_path(f"models/staging_models/{date}/tokenizer_config.json"), "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
    sequences=tokenizer.texts_to_sequences(X['description'])
    padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")
    # Prediction
    lstm_proba=model.predict([padded_sequences])
    final_prediction = np.argmax(lstm_proba)   
    return final_prediction

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)        
    img_array = preprocess_input(img_array)
    return img_array


def image_predict(X,model,date):
    """
    Returns : [prdtypecode,sequence of probabilities]
    """
    images=X['image_path'].apply(lambda x : preprocess_image(x,target_size=(224,224,3)))
    images=tf.convert_to_tensor(images.tolist(),dtype=tf.float32)

    # Prediction 
    vgg16_proba = model.predict([images]) 
    final_prediction = np.argmax(vgg16_proba)
    # Returns the prdtypecode and the probabilities for the 27 categories
    return vgg16_proba

def concatenate_predict(lstm_r,vgg16_r,best_weights):
    concatenate_proba = (best_weights[0] * lstm_r + best_weights[1] * vgg16_r)
    final_prediction = np.argmax(concatenate_proba,axis=1)
    return [int(mapper[str(final_prediction[i])]) for i in range(len(final_prediction))] 

def main(date):
    df=pd.read_csv(resolve_path(f"models/staging_models/{date}/final_test.csv"))
    print(df.head(5))
    print('A')
    # Loads the trained models for the text and image predictions
    vgg16_model=keras.models.load_model(resolve_path(f"models/staging_models/{date}/best_vgg16_model.keras"))
    lstm_model=keras.models.load_model(resolve_path(f"models/staging_models/{date}/best_lstm_model.keras"))
    print('B')
    best_weights=None
    # Loads the weights for the concatenation of the results of the previous models
    with open(resolve_path(f"models/staging_models/{date}/best_weights.pkl"),"rb") as file :
        best_weights=pickle.load(file)
    print('C')    
    response=None
    lstm_return=text_predict(df,lstm_model,date)
    print('C1')
    vgg16_return=image_predict(df,vgg16_model,date)
    print('D')
    response=concatenate_predict(lstm_r=lstm_return,vgg16_r=vgg16_return,best_weights=best_weights)
    print(response)
    #print(df['prdtypecode'].tolist())
    print('E')
    y=[int(mapper[str(df['prdtypecode'][i])]) for i in range(len(df['prdtypecode']))]
    print(y)
    print(len(response))
    print(len(y))
    print(precision_score(response,y,average='weighted'))
    
if __name__=="__main__":
    main('20240706_00-55-27')