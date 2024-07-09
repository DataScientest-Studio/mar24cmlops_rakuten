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
from utils.make_db import download_initial_db
from utils.s3_utils import create_s3_conn_from_creds, download_from_s3
from utils.resolve_path import resolve_path
from concurrent.futures import ThreadPoolExecutor
import duckdb
import pickle
import json
import os
import re
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv('.env/.env.development')
        
aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
duckdb_path = os.path.join(resolve_path(os.environ['DATA_PATH']), os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']

class production_model_retrain:
    def __init__(self,model_type='production',version='latest',model_name=None):
        self.model_type=model_type
        self.version=version
        self.model_name=model_name
        self.lemmatizer = None
        self.tokenizer = None
        self.stop_words = None
        self.lstm = None
        self.vgg16 = None
        self.mapper = None
        self.best_weights = None
        self.Xy=[]
        self.load_txt_utils()
        self.load_img_utils()
        self.load_model_utils()    
        self.load_duckdb_utils()    
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        
    def get_model_path(self):
        if self.model_type == 'production':
            folder = 'production_model'
        else:
            folder = 'staging_models'
            if self.version == 'latest':
                versions = sorted(os.listdir(base_path))
                version = versions[-1]
                
                base_path = resolve_path(f'models/{folder}/{version}/')
        base_path = resolve_path(f'models/{folder}/')
        
        return base_path
    
    def load_txt_utils(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))
        
        model_path = self.get_model_path()
        print(model_path)
        with open(os.path.join(model_path, "tokenizer_config.json"), "r", encoding="utf-8") as json_file:
            tokenizer_config = json_file.read()
        self.tokenizer = tokenizer_from_json(tokenizer_config)
        self.lstm = keras.models.load_model(os.path.join(model_path, "best_lstm_model.keras"))
    
    def load_img_utils(self):
        model_path = self.get_model_path()
        self.vgg16 = keras.models.load_model(os.path.join(model_path, "best_vgg16_model.keras"))
    
    def load_model_utils(self):
        model_path = self.get_model_path()
        
        with open(os.path.join(model_path, "mapper.json"), "r") as json_file:
            self.mapper = json.load(json_file)
            
        with open(os.path.join(model_path, "best_weights.pkl"), "rb") as file:
            self.best_weights = pickle.load(file)
    
    def load_duckdb_utils(self):
        self.db_conn=duckdb.connect(database=duckdb_path, read_only=False)
    
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
            padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")
            return padded_sequences
        return None
    
    def path_to_img(self, img_path):
        img = load_img(img_path, target_size=(224, 224, 3))
        return img

    def byte_to_img(self, file):
        img = load_img(BytesIO(file), target_size=(224, 224, 3))
        return img    
    
    def process_img(self, img):
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict_text(self, text_sequence):
        probability = self.lstm.predict([text_sequence])
        pred = np.argmax(probability)   
        return int(self.mapper[str(pred)]), probability
    
    def predict_img(self, img_array):
        images = tf.convert_to_tensor([img_array], dtype=tf.float32)
        probability = self.vgg16.predict([images]) 
        pred = np.argmax(probability)
        return int(self.mapper[str(pred)]), probability
    
    def agg_prediction(self, txt_prob, img_prob):
        concatenate_proba = (self.best_weights[0] * txt_prob + self.best_weights[1] * img_prob)
        pred = np.argmax(concatenate_proba)
        return int(self.mapper[str(pred)]) 
    
    def predict(self, designation, image):
        text_sequence = self.process_txt(designation)
        if isinstance(image, str):  # If image is a path
            img = self.path_to_img(image)
        elif isinstance(image, bytes):  # If image is bytes
            img = self.byte_to_img(image)
        else:
            raise ValueError("Image must be a file path or bytes.")
        
        img_array = self.process_img(img)
        
        _, txt_prob = self.predict_text(text_sequence)
        _, img_prob = self.predict_img(img_array)
        
        agg_pred = self.agg_prediction(txt_prob, img_prob)
        return agg_pred        
        
        
    def split_train_test(self,df,train_ratio=5/7.,test_ratio=1/7.,test_val=1/7.,limit_train=27*600,limit_test=27*50,limit_val=27*50):
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
        X=df.drop(['prdtypecode'],axis=1)
        y=df['prdtypecode']
        X_train1_init,X_test1,y_train1_init,y_test1=train_test_split(X,y,test_size=test_ratio,random_state=42,stratify=y)    
        X_train1,X_val1,y_train1,y_val1=train_test_split(X_train1_init,y_train1_init,test_size=test_ratio/(train_ratio+test_ratio),random_state=42,stratify=y_train1_init)
        X_train,y_train=RandomOverSampler().fit_resample(X_train1,y_train1)
        X_test,y_test=RandomOverSampler().fit_resample(X_test1,y_test1)
        X_val,y_val=RandomOverSampler().fit_resample(X_val1,y_val1)
        return [[X_train[:limit_train],y_train[:limit_train]],[X_val[:limit_val],y_val[:limit_val]],[X_test[:limit_test],y_test[:limit_test]]]

    def detection_sousdossier(self,image_path):
        for i in ['image_train','temporary_image','image_test']: ##################### à modifier
            if os.path.isfile(f"data/preprocessed/{i}/{os.path.basename(image_path)}"):
                return f"data/preprocessed/{i}/{os.path.basename(image_path)}"


    def data_handle(self,db_limitation=False,init=False):
        liste_valeur=None
        if init is False:
            if db_limitation is True:
                liste_valeur=self.db_conn.sql("SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user IS NOT NULL AND NOT user='init_user' AND user_prdtypecode IS NOT NULL;").df()
            else:
                liste_valeur=self.db_conn.sql('SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user IS NOT NULL AND user_prdtypecode IS NOT NULL;').df()
        else :
            liste_valeur=self.db_conn.sql("SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user='init_user';").df() 
        print(liste_valeur[:4]) ##################
        mapper_inv={self.mapper[i]:i for i in self.mapper.keys()}
        liste_valeur['prdtypecode']=liste_valeur['prdtypecode'].astype(str).replace(mapper_inv)
        liste_valeur['description']=liste_valeur['designation']+str(liste_valeur['description'])
        liste_valeur.drop(['designation'],axis=1,inplace=True)
        filepath='data/preprocessed/image_train' # limitation à un seul dossier
        # resolve_path ?
        liste_valeur["image_path"] = liste_valeur.apply(lambda row : f"{filepath}/image_{row['imageid']}_product_{row['productid']}.jpg",axis=1)
        #liste_valeur['description']=liste_valeur['description'].apply(self.process_txt)
        liste_valeur.dropna(axis=0,subset=['image_path','description'],how='any',inplace=True) ## et description ???
        self.Xy = self.split_train_test(liste_valeur)
        print(self.Xy[0][0][:2]) ######################
        print(self.Xy[0][1][:2]) ######################
        print(self.Xy[1][0][:2]) ######################
        print(self.Xy[1][1][:2]) ######################
        print(self.Xy[2][0][:2]) ######################
        print(self.Xy[2][1][:2]) ######################        
    
    def preprocess_txt(self, text):
        if text is not None:
            text = re.sub(r"[^a-zA-Z]", " ", text)
            words = word_tokenize(text.lower())
            filtered_words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words and word!='none']    ########################       
            textes = " ".join(filtered_words[:10])
            return textes
        return None
    
    def tokenize(self,path):
        #liste=[x for x in self.Xy[0][0]['description'] if len(x)!=0]
        print(self.Xy[0][0]['description'][:5])
        self.tokenizer.fit_on_texts(self.Xy[0][0]['description'])
        #self.tokenizer.fit_on_texts(liste)        

        os.exit()
        tokenizer_config = self.tokenizer.to_json()
        with open(os.path.join(path,'tokenizer_config.json'),'w') as file:
            file.write(tokenizer_config)
    
    def text_train(self,is_production,epochs=1):  #############" paramètres ?"
        print(self.Xy[0][0]['description'][:10])  ####################
        self.Xy[0][0]['description']=self.Xy[0][0]['description'].apply(lambda text : self.preprocess_txt(text)) 
        print(self.Xy[0][0]['description'][:11])  ####################        
        # problème : il reste des non-alphanum.
        self.tokenize(path=self.get_model_path())
        print(self.Xy[0][0]['description'][:10]) ##############
        train_sequences = self.tokenizer.texts_to_sequences(self.Xy[0][0]["description"])
        train_padded_sequences = pad_sequences(train_sequences, maxlen=10, padding="post", truncating="post")
         #os.exit()
        self.Xy[1][0]['description']=self.Xy[1][0]['description'].apply(lambda text : self.preprocess_txt(text))
        val_sequences = self.tokenizer.texts_to_sequences(self.Xy[1][0]["description"])
        val_padded_sequences = pad_sequences(val_sequences, maxlen=10, padding="post", truncating="post")
        
        text_input = Input(shape=(self.max_sequence_length,))
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(
            text_input
        )
        ### Moduler içi pour mettre le bon modèle (production ou non)
        lstm_layer = LSTM(128)(embedding_layer)
        output = Dense(27, activation="softmax")(lstm_layer)

        self.lstm = Model(inputs=[text_input], outputs=output)

        if is_Production is True:
            self.lstm=keras.models.load_model(os.path.join(prefix,"models/production_model/best_lstm_model.keras")) 

        self.lstm.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        
        lstm_callbacks = [
            ModelCheckpoint(
                filepath=resolve_path(f"models/staging_models/{date_path}/best_lstm_model.keras"), save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.lstm.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(self.Xy[0][1], num_classes=27),
            epochs=epochs,
            batch_size=32,
            validation_data=(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(self.Xy[1][1], num_classes=27),
            ),
            callbacks=lstm_callbacks,
        )
                
    def image_train(self,path,is_production=True,epochs=1):  ######### Arguments
        batch_size = 32
        num_classes = 27

        df_train = pd.concat([self.Xy[0][0], self.Xy[0][1].astype(str)], axis=1)
        df_val = pd.concat([self.Xy[1][0], self.Xy[1][1].astype(str)], axis=1)

        # Créer un générateur d'images pour le set d'entraînement
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Normalisation des valeurs de pixel
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
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Normalisation des valeurs de pixel
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,  # Pas de mélange pour le set de validation
        )
    
        image_input = Input(shape=(224, 224, 3))  # Adjust input shape according to your images

        vgg16_base = VGG16(include_top=False, weights="imagenet", input_tensor=image_input)

        x = vgg16_base.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)  # Add some additional layers if needed
        output = Dense(num_classes, activation="softmax")(x)

        self.vgg16 = Model(inputs=vgg16_base.input, outputs=output)

        for layer in vgg16_base.layers:
            layer.trainable = False

        if is_Production is True:
            self.vgg16=keras.models.load_model(os.path.join(prefix,"models/production_model/best_vgg16_model.keras"))

        self.vgg16.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        vgg_callbacks = [
            ModelCheckpoint(
                filepath=resolve_path(f"models/staging_models/{date_path}/best_vgg16_model.keras"), save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.vgg16.fit(
            train_generator,
            epochs=epochs,              ############# le temps de l'évaluation
            validation_data=val_generator,
            callbacks=vgg_callbacks,
        )
    
    def concatenate_train(self):
        pass
    
    
    
    def train(self,is_production=True,epochs=1):
        self.text_train(is_production=is_production, epochs=epochs)
        #self.image_train()
        #self.concatenate_train()
        # sauvegarde models
        # enregistrement sauvegarde scores