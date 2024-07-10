import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pickle
import json
from datetime import datetime
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import duckdb
from utils.make_db import download_initial_db
from utils.s3_utils import create_s3_conn_from_creds, download_from_s3
from concurrent.futures import ThreadPoolExecutor
from utils.resolve_path import resolve_path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load environment variables from .env file
load_dotenv(resolve_path('.env/.env.development'))
        
aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
duckdb_path = os.path.join(resolve_path(os.environ['DATA_PATH']), os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']

# Check if the DuckDB database file exists locally, if not, download it from S3
if not os.path.isfile(duckdb_path):
    print('No Database Found locally')
    # Since no database found for the API, download the initial database from S3
    download_initial_db(aws_config_path, duckdb_path)
    print('Database Sucessfully Downloaded')

# Download database for the mapping of the results    
db_conn = duckdb.connect(database=duckdb_path, read_only=False)

prd_categories=dict()
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
with open(os.path.join(prefix,"models/mapper.json"), "r") as json_file:   #
    mapper = json.load(json_file)
    
def split_train_test(df,train_ratio=5/7.,test_ratio=1/7.,test_val=1/7.,limit_train=27*600,limit_test=27*50,limit_val=27*50):
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
    return X_train[:limit_train],X_val[:limit_val],X_test[:limit_test],y_train[:limit_train],y_val[:limit_val],y_test[:limit_test]

def preprocess_text(text):
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

class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val,date_path,is_Production=False,epochs=1):
        ##############
        print('coucou') ############################
        print(X_train["description"][:5]) ##########################
        ##############
        self.tokenizer.fit_on_texts(X_train["description"])

        tokenizer_config = self.tokenizer.to_json()
        with open(resolve_path(f"models/staging_models/{date_path}/tokenizer_config.json"), "w", encoding="utf-8") as json_file:
            json_file.write(tokenizer_config)

        train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

        val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
        val_padded_sequences = pad_sequences(
            val_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

        text_input = Input(shape=(self.max_sequence_length,))
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(
            text_input
        )
        lstm_layer = LSTM(128)(embedding_layer)
        output = Dense(27, activation="softmax")(lstm_layer)

        self.model = Model(inputs=[text_input], outputs=output)

        if is_Production is True:
            self.model=keras.models.load_model(os.path.join(prefix,"models/production_model/best_lstm_model.keras")) 

        self.model.compile(
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

        self.model.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(y_train, num_classes=27),
            epochs=epochs,
            batch_size=32,
            validation_data=(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(y_val, num_classes=27),
            ),
            callbacks=lstm_callbacks,
        )

class ImageVGG16Model:
    def __init__(self):
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val,date_path,is_Production=False,epochs=1):
        # Paramètres
        batch_size = 32
        num_classes = 27

        df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
        df_val = pd.concat([X_val, y_val.astype(str)], axis=1)

        # Créer un générateur d'images pour le set d'entraînement
        train_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
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
        val_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,  # Pas de mélange pour le set de validation
        )

        image_input = Input(
            shape=(224, 224, 3)
        )  # Adjust input shape according to your images

        vgg16_base = VGG16(
            include_top=False, weights="imagenet", input_tensor=image_input
        )

        x = vgg16_base.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)  # Add some additional layers if needed
        output = Dense(num_classes, activation="softmax")(x)

        self.model = Model(inputs=vgg16_base.input, outputs=output)

        for layer in vgg16_base.layers:
            layer.trainable = False

        if is_Production is True:
            self.model=keras.models.load_model(os.path.join(prefix,"models/production_model/best_vgg16_model.keras"))

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        vgg_callbacks = [
            ModelCheckpoint(
                filepath=resolve_path(f"models/staging_models/{date_path}/best_vgg16_model.keras"), save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.model.fit(
            train_generator,
            epochs=epochs,              ############# le temps de l'évaluation
            validation_data=val_generator,
            callbacks=vgg_callbacks,
        )


class concatenate:
    def __init__(self, tokenizer, lstm, vgg16):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(
        self, X_train, y_train, new_samples_per_class=50, max_sequence_length=10
    ):
        num_classes = 27

        new_X_train = pd.DataFrame(columns=X_train.columns)
        new_y_train = pd.DataFrame(
            columns=[0]
        )  # Créez la structure pour les étiquettes

        # Boucle à travers chaque classe
        for class_label in range(num_classes):
            # Indices des échantillons appartenant à la classe actuelle
            indices = np.where(y_train == str(class_label))[0]

            # Sous-échantillonnage aléatoire pour sélectionner 'new_samples_per_class' échantillons
            sampled_indices = resample(
                indices, n_samples=new_samples_per_class, replace=False, random_state=42
            )

            # Ajout des échantillons sous-échantillonnés et de leurs étiquettes aux DataFrames
            new_X_train = pd.concat([new_X_train, X_train.loc[sampled_indices]])
            new_y_train = pd.concat([new_y_train, y_train.loc[sampled_indices]])

        # Réinitialiser les index des DataFrames
        new_X_train = new_X_train.reset_index(drop=True)
        print(new_X_train.head(5))
        new_y_train = new_y_train.reset_index(drop=True)[['prdtypecode']]
        
        print(new_y_train.head(5))
        new_y_train = new_y_train.values.reshape(1350).astype("int")

        # Charger les modèles préalablement sauvegardés
        tokenizer = self.tokenizer
        lstm_model = self.lstm
        vgg16_model = self.vgg16

        train_sequences = tokenizer.texts_to_sequences(new_X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences, maxlen=10, padding="post", truncating="post"
        )

        # Paramètres pour le prétraitement des images
        target_size = (224,224,3) 

        images_train = new_X_train["image_path"].apply(lambda x: self.preprocess_image(x, target_size))

        images_train = tf.convert_to_tensor(images_train.tolist(), dtype=tf.float32)

        lstm_proba = lstm_model.predict([train_padded_sequences])

        vgg16_proba = vgg16_model.predict([images_train])

        return lstm_proba, vgg16_proba, new_y_train

    def optimize(self, lstm_proba, vgg16_proba, y_train):
        # Recherche des poids optimaux en utilisant la validation croisée
        best_weights = None
        best_accuracy = 0.0

        for lstm_weight in np.linspace(0, 1, 101):  # Essayer différents poids pour LSTM
            vgg16_weight = 1.0 - lstm_weight  # Le poids total doit être égal à 1

            combined_predictions = (lstm_weight * lstm_proba) + (
                vgg16_weight * vgg16_proba
            )
            final_predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(y_train, final_predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (lstm_weight, vgg16_weight)

        return best_weights

def detection_sousdossier(image_path):
    for i in ['image_train','temporary_image','image_test']:
        if os.path.isfile(f"data/preprocessed/{i}/{os.path.basename(image_path)}"):
            return f"data/preprocessed/{i}/{os.path.basename(image_path)}"


def retrain(is_Production=False,db_limitation=False,init=False,epochs=1):
    image_vgg16_model=ImageVGG16Model()
    text_lstm_model=TextLSTMModel()
    
    date_stage=datetime.now().strftime('%Y%m%d_%H-%M-%S')
    print(date_stage)
    if not os.path.isdir(f"models/staging_models/{date_stage}/"):
        os.mkdir(f"models/staging_models/{date_stage}/")
                #db_conn = duckdb.connect(database=duckdb_path, read_only=False)
    liste_valeur=None
    if init is False:
        if db_limitation is True:
            liste_valeur=db_conn.sql("SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user IS NOT NULL AND NOT user='init_user' AND user_prdtypecode IS NOT NULL;").df()
        else:
            liste_valeur=db_conn.sql('SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user IS NOT NULL AND user_prdtypecode IS NOT NULL;').df()
    else :
        liste_valeur=db_conn.sql("SELECT designation,description,imageid,productid,user_prdtypecode AS prdtypecode FROM fact_listings WHERE user='init_user';").df() 

    mapper_inv={mapper[i]:i for i in mapper.keys()}
    liste_valeur['prdtypecode']=liste_valeur['prdtypecode'].astype(str)
    liste_valeur['prdtypecode']=liste_valeur['prdtypecode'].replace(mapper_inv)
    liste_valeur['description']=liste_valeur['designation']+str(liste_valeur['description'])
    liste_valeur.drop(['designation'],axis=1,inplace=True)
    filepath='image_train'
    liste_valeur["image_path"] = liste_valeur.apply(lambda row : f"{filepath}/image_{row['imageid']}_product_{row['productid']}.jpg",axis=1)
    liste_valeur['description']=liste_valeur['description'].apply(preprocess_text)
    #print(liste_valeur['description'][:10]) ########################
    liste_valeur['image_path']=liste_valeur['image_path'].apply(lambda x : detection_sousdossier(x))
    liste_valeur.dropna(axis=0,subset=['image_path'],how='any',inplace=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(liste_valeur)
    if not os.path.isdir(resolve_path('data/preprocessed/temporary_images')):
        os.mkdir(resolve_path('data/preprocessed/temporary_images'))
    
    liste_local_image=[]
    for _,a,file_name in os.walk(resolve_path('data/preprocessed')):
        liste_local_image=liste_local_image+file_name
    client = create_s3_conn_from_creds(resolve_path(os.getenv('AWS_CONFIG_PATH')))
    liste_image=list(X_train['image_path'])+list(X_test['image_path'])+list(X_val['image_path'])
            #liste_s3_image=[i for i in liste_image if os.path.basename(i) not in liste_local_image]
                #liste_s3_image=[os.path.basename(i) for i in liste_local_image]
                #print(liste_s3_image)
                #return
                #liste_local_image=[os.path.join(resolve_path('data/temporary'),os.path.basename(i))  for i in liste_s3_image]
                #liste_local_image1=[os.path.join('data/preprocessed/temporary_images',os.path.basename(i)) for i in liste_s3_image]
            #with ThreadPoolExecutor() as executor:
            #    executor.map(lambda bucket_x_local: download_from_s3(client,bucket_x_local[0],bucket_x_local[1]), zip(liste_s3_image,liste_local_image1))
            #for i in liste_s3_image:    
            #    download_from_s3(client,i,os.path.join('data/preprocessed/temporary_images',os.path.basename(i)))
                #    print(i)
                
                #return
    # Train LSTM model
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)
    print("X_val: ", X_val.shape)
    print("y_val: ", y_val.shape)
    print(np.unique(y_train))
    print("Training LSTM Model")
    text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val,date_stage,is_Production,epochs)
    print("Finished training LSTM")
    #return
    print("Training VGG")
    # Train VGG16 model
    image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val,date_stage,is_Production,epochs)
    print("Finished training VGG")
    #return
    with open(resolve_path(f"models/staging_models/{date_stage}/tokenizer_config.json"), "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
    lstm = keras.models.load_model(resolve_path(f"models/staging_models/{date_stage}/best_lstm_model.keras"))
    vgg16 = keras.models.load_model(resolve_path(f"models/staging_models/{date_stage}/best_vgg16_model.keras"))

    print("Training the concatenate model")
    model_concatenate = concatenate(tokenizer, lstm, vgg16)
    lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
    best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
    print("Finished training concatenate model")

    with open(resolve_path(f"models/staging_models/{date_stage}/best_weights.pkl"), "wb") as file:
        pickle.dump(best_weights, file)

    num_classes = 27

    proba_lstm = keras.layers.Input(shape=(num_classes,))
    proba_vgg16 = keras.layers.Input(shape=(num_classes,))

    weighted_proba = keras.layers.Lambda(
        lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
    )([proba_lstm, proba_vgg16])

    concatenate_model = keras.models.Model(
        inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
    )

    # Enregistrer le modèle au format h5
    concatenate_model.save(resolve_path(f"models/staging_models/{date_stage}/concatenate.keras"))
    
    # Recording of X_test, y_test for future comparison
    final_test=X_test
    final_test['prdtypecode']=y_test
    final_test.to_csv(resolve_path(f"models/staging_models/{date_stage}/final_test.csv"))

if __name__ == '__main__':
    #retrain(is_Production=False, db_limitation=False)
    #retrain(is_Production=True, db_limitation=False)
    #retrain(is_Production=False, db_limitation=True)
    #retrain(is_Production=True, db_limitation=True)
    #retrain(is_Production=False, db_limitation=False,init=True)
    retrain(is_Production=False, db_limitation=False,init=True,epochs=2)