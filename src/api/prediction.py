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


s3_client=None
db_conn=None
prd_categories=dict()

# Load environment variables from .env file
load_dotenv('.env/.env.development')
        
aws_config_path = os.environ['AWS_CONFIG_PATH']
duckdb_path = os.path.join(os.environ['DATA_PATH'], os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']

# Download database for the mapping of the results    
db_conn = duckdb.connect(database=duckdb_path, read_only=False)
categories=db_conn.sql('SELECT * FROM dim_prdtypecode').fetchall()    
for i in categories:
    prd_categories[i[1]]=i[2]

# Loads every context to allow text_analysis
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("french"))

# Modification of the pathways following the environnment
prefix=None
if not os.getenv('IS_CONTAINER'):
    prefix='.'
else :
    prefix='/app'
# Loads the mapper between the return of the models and the associated prdtypecode
with open(os.path.join(prefix,"models/mapper.json"), "r") as json_file:
    mapper = json.load(json_file)

# Loads the trained models for the text and image predictions
vgg16=keras.models.load_model(os.path.join(prefix,"models/production_model/best_vgg16_model.keras"))
lstm=keras.models.load_model(os.path.join(prefix,"models/production_model/best_lstm_model.keras"))
best_weights=None

# Loads the weights for the concatenation of the results of the previous models
with open(os.path.join(prefix,"models/production_model/best_weights.pkl"),"rb") as file :
    best_weights=pickle.load(file)

def change_model(pathway : str = None):  # A tester
    """
    Function to call if the model to work with is not a production model : 
    allows the actualization of the vgg16, lstm and best_weights variables
    (by default, actualizes with the model in production)
    Argument:
        pathway : name of the folder in models/staging where the model resides
    """
    global vgg16, lstm, best_weights  # Non nécessaire si ce sont des objets
    if pathway is not None:
        # Verifies the existence of the model and downloads it else raise an exceptio
        if os.path.isdir(f"models/staging_models/{pathway}"):
            vgg16=keras.models.load_model(os.path.join(prefix,f"models/staging_models/{pathway}/best_vgg16_model.keras"))
            lstm=keras.models.load_model(os.path.join(prefix,f"models/staging_models/{pathway}/best_lstm_model.keras"))
            with open(os.path.join(prefix,f"models/staging_models/{pathway}/best_weights.pkl"),"rb") as file :
                best_weights=pickle.load(file)
        else:
            raise Exception("Bad location for a model")
    # By default, without any argument, uses the model in production
    else :
        vgg16=keras.models.load_model(os.path.join(prefix,"models/production_model/best_vgg16_model.keras"))
        lstm=keras.models.load_model(os.path.join(prefix,"models/production_model/best_lstm_model.keras"))
        with open("models/best_weights.pkl","rb") as file :
            best_weights=pickle.load(file)


def text_predict(text):
    """
    Text prediction
    Argument:
        entry : the text sequence to evaluate
    Returns:
        [prdtypecode,sequence of probabilities]
     """
    if text is not None:
        with open(os.path.join(prefix,"models/tokenizer_config.json"), "r", encoding="utf-8") as json_file:
            tokenizer_config = json_file.read()
        tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
        # Removes all the non-alphabetic characters
        text = re.sub(r"[^a-zA-Z]", " ", text)
        # Tokenization
        words = word_tokenize(text.lower())
        # Removes the stop wordds and lemmatization
        filtered_words = [
            lemmatizer.lemmatize(word)
            for word in words
            if word not in stop_words
            ]
        # The tenth most frequent words as a string
        textes = " ".join(filtered_words[:10])
        # Tokenization and formatting of the tokens
        sequences=tokenizer.texts_to_sequences([textes])
        padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")
        # Prediction
        lstm_proba=lstm.predict([padded_sequences])
        final_prediction = np.argmax(lstm_proba)   
        return [int(mapper[str(final_prediction)]),lstm_proba]
    # Returns None if no text provided
    return None

def image_predict(s3_client = None, imageid : int = None, productid : int = None, directory : str = 'image_train', new_image : str = None):
    """
    Image prediction following 2 ways :
        _with the directory where the image named following imageid and productid is stored
        _with a complete new pathway, name included
    Returns : [prdtypecode,sequence of probabilities]
    """
    img=None
    imagepath=None
    # if imageid and productid specified
    if imageid is not None and productid is not None and s3_client is not None :
        imagepath=os.path.join(directory,f"image_{imageid}_product_{productid}.jpg")
        print(imagepath)
        s3_file=BytesIO()
        s3_client.download_fileobj("rakutenprojectbucket",imagepath,s3_file)
        s3_file.seek(0)
        img=load_img(s3_file,target_size=(224,224,3))
    # if specified, new_image has the priority over imageid, productid and directory
    if new_image is not None:
        imagepath=new_image
        img=load_img(imagepath,target_size=(224,224,3))
    # Loading of the image and preprocessing following vgg16
    
    img_array = img_to_array(img)        
    img_array = preprocess_input(img_array)
    images = tf.convert_to_tensor([img_array], dtype=tf.float32)
    # Prediction 
    vgg16_proba = vgg16.predict([images]) 
    final_prediction = np.argmax(vgg16_proba)
    # Returns the prdtypecode and the probabilities for the 27 categories
    return [int(mapper[str(final_prediction)]),vgg16_proba]

def image_object_predict(img):
    """
    Image prediction.
    Arguments :
        picture : a PIL format image object
    Returns :
        [prdtypecode,sequence of probabilities]
    """
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    images = tf.convert_to_tensor([img_array], dtype=tf.float32)
    vgg16_proba = vgg16.predict([images]) 
    final_prediction = np.argmax(vgg16_proba)
    return [int(mapper[str(final_prediction)]),vgg16_proba]

def repartition_prediction(s3_client = None, designation : str =None, imageid : int = None, productid : int = None, directory : str = 'image_train', new_image : str = None, file = None):
    lstm_return=None
    vgg16_return=None
    img=None
    # text prediction if designation specified
    if designation is not None :
        lstm_return=text_predict(designation)
    # image prediction if (imageid and productid) or new_image provided
    if imageid is not None and productid is not None and s3_client is not None:
        vgg16_return=image_predict(s3_client=s3_client, imageid=imageid,productid=productid,directory=directory)
    elif new_image is not None:
        vgg16_return=image_predict(new_image=new_image)
    # image prediction if image object (file) provided
    if file is not None:
        img=load_img(BytesIO(file),target_size=(224,224,3))
        vgg16_return=image_object_predict(img)
    # Return prdtypecode prediction if only designation provided
    if lstm_return is not None and vgg16_return is None:
        return int(lstm_return[0])
    # Returns prdtypecode prediction if only image provided 
    if vgg16_return is not None and lstm_return is None:
        return int(vgg16_return[0])
    # Prediction if both image and designation provided
    if lstm_return is not None and vgg16 is not None:
        concatenate_proba = (best_weights[0] * lstm_return[1] + best_weights[1] * vgg16_return[1])
        final_prediction = np.argmax(concatenate_proba)
        return int(mapper[str(final_prediction)]) 
    raise Exception("La prédiction n\'a pas pu aboutir.")

def concatenate_predict(lstm_r,vgg16_r):
    concatenate_proba = (best_weights[0] * lstm_r + best_weights[1] * vgg16_r)
    final_prediction = np.argmax(concatenate_proba)
    return int(mapper[str(final_prediction)]) 

def image_management(s3_client = None, imageid : int = None, productid : int = None, directory : str = 'image_train', new_image : str = None, file = None):
    """
    Manage the image part of the request with the priority : 
    image object >> new_image >> (imageid,productid)

    Args:
        s3_client (_type_, optional): _description_. Defaults to None.
        imageid (int, optional): _description_. Defaults to None.
        productid (int, optional): _description_. Defaults to None.
        directory (str, optional): _description_. Defaults to 'image_train'.
        new_image (str, optional): _description_. Defaults to None.
        file (_type_, optional): _description_. Defaults to None.
    Returns:
        the prdtypecode
    """
    # image prediction if image object (file) provided
    if file is not None:
        img=load_img(BytesIO(file),target_size=(224,224,3))
        vgg16_return=image_object_predict(img)
        return vgg16_return
    # image prediction if new_image (pathway+filename) provided
    if new_image is not None:
        vgg16_return=image_predict(new_image=new_image)
        return vgg16_return
    # image prediction if imageid and productid and directory provided
    if imageid is not None and productid is not None and s3_client is not None:
        vgg16_return=image_predict(s3_client=s3_client, imageid=imageid,productid=productid,directory=directory)
        return vgg16_return
    # Returns None if no image found
    return None
    
    
def predict_with_unified_interface(s3_client = None, designation : str =None, imageid : int = None, productid : int = None, directory : str = 'image_train', new_image : str = None, file = None):
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
    response=None
    lstm_return=text_predict(designation)
    vgg16_return=image_management(s3_client=s3_client, imageid=imageid, productid=productid, directory=directory, new_image=new_image, file=file)
    if lstm_return is not None and vgg16_return is None:
        response=lstm_return[0]
    if lstm_return is None and vgg16_return is not None:
        response=vgg16_return[0]
    if lstm_return is not None and vgg16_return is not None:
        response=concatenate_predict(lstm_r=lstm_return[1],vgg16_r=vgg16_return[1])
    print(response)
    print(type(response))
    if response is not None:
        return prd_categories[response]
    else:
        raise Exception('Prediction failed')


def main():
    """
    The aim is to test the efficiency of this module by various examples
    representing the various possibilities of arguments
    """

    
    client = create_s3_conn_from_creds(os.getenv('AWS_CONFIG_PATH'))
    data=BytesIO()
    client.download_fileobj("rakutenprojectbucket","image_train/image_1007738129_product_435130019.jpg",data)
#    with open('temp.jpg','wb') as f:    
#        client.download_fileobj("rakutenprojectbucket","image_train/image_1007738129_product_435130019.jpg",f)
#    img=load_img('temp.jpg')
    data.seek(0)
    img=load_img(data)
#    print(type(img))
#    print(len(img))
    #img=img.tobytes()
    #img=data
    #save_img('coucou3.jpg',img_to_array(img,dtype=int))
#    download_from_s3(s3_conn=s3_client, 
#                 bucket_path ='image_train/image_1007738129_product_435130019.jpg',
#                 local_path =os.path.join(os.getcwd(),'coucou1.jpg'))

    # Text prediction
    print(predict_with_unified_interface(designation="import jeu video japon"))
    print(predict_with_unified_interface(designation="reine roumanie paris"))
    # Recorded image prediction
    #print(predict_with_unified_interface(imageid=234234,productid=184251,directory=os.path.join(prefix,"data/preprocessed/image_train")))
    # Prediction with both recorded image and text
    #print(predict_with_unified_interface(imageid=234234,productid=184251,directory=os.path.join(prefix,"data/preprocessed/image_train"),designation="import jeu video japon"))
    # Prediction with a chunk of bytes representing an image in memory
    #img=open(os.path.join(prefix,'data/preprocessed/image_train/image_234234_product_184251.jpg'),'rb').read()
    #print(predict_with_unified_interface(file=img))
    # Prediction with both a designation and a chunk of bytes representing an image in memory
    #img=open(os.path.join(prefix,'data/preprocessed/image_train/image_234234_product_184251.jpg'),'rb').read()
    #print(predict_with_unified_interface(file=img, designation="reine roumanie paris"))
    print(predict_with_unified_interface(new_image='temp.jpg', designation="reine roumanie paris"))
    print(predict_with_unified_interface(new_image='temp.jpg'))
    print(predict_with_unified_interface(s3_client=client, imageid=1007738129,productid=435130019))
    
if __name__=="__main__":
    main()
