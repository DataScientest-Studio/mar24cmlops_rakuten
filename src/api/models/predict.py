import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
import os 
from api.utils.resolve_path import resolve_path
import re 
import numpy as np
import json
from io import BytesIO
import pickle

# Une fonction de preprocessing, une fonction de prediction,
# une fonction de chargement de modele

def load_txt_utils():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("french"))
    
    with open(os.path.join(resolve_path('models/production_model/'),"tokenizer_config.json"), "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_config)
    lstm = keras.models.load_model(os.path.join(resolve_path('models/production_model/'),"best_lstm_model.keras"))
    
    return lemmatizer, tokenizer, stop_words, lstm

def load_img_utils():
    vgg16 = keras.models.load_model(os.path.join(resolve_path('models/production_model/'),"best_vgg16_model.keras"))
    return vgg16

def load_model_utils():
    
    with open(os.path.join(resolve_path('models/production_model/'),"mapper.json"), "r") as json_file:
        mapper = json.load(json_file)
        
    with open(os.path.join(resolve_path('models/production_model/'),"best_weights.pkl"),"rb") as file :
        best_weights=pickle.load(file)
        
    return mapper, best_weights

def process_txt(text,tokenizer):
    
    if text is not None:
        # Removes all the non-alphabetic characters
        text = re.sub(r"[^a-zA-Z]", " ", text)
        # Tokenization
        words = word_tokenize(text.lower())
        # Removes the stop words and lemmatization
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
        
        return padded_sequences
    
    return None

def path_to_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    return img

def byte_to_img(file):
    img=load_img(BytesIO(file),target_size=(224,224,3))
    return img

def process_img(img):
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def predict_text(model, text_sequence, mapper):
    probability = model.predict([text_sequence])
    pred = np.argmax(probability)   
    return int(mapper[str(pred)]), probability

def predict_img(model, img_array, mapper):
    images = tf.convert_to_tensor([img_array], dtype=tf.float32)
    probability = model.predict([images]) 
    pred = np.argmax(probability)
    return int(mapper[str(pred)]), probability

def agg_prediction(txt_prob, img_prob, mapper, best_weights):
    concatenate_proba = (best_weights[0] * txt_prob + best_weights[1] * img_prob)
    pred = np.argmax(concatenate_proba)
    return int(mapper[str(pred)]) 

def load_model(is_production = True):
    pass
    #txt_model, img_model, concat_model = None
    #return txt_model, img_model, concat_model
    
def predict_existing_listing():
    pass

def predict(designation, image, txt_model, img_model, tokenizer):
    
    text_sequence = process_txt(designation,tokenizer)
    img_array = 
    
    _, txt_prob = predict_text(txt_model, text_sequence, mapper)
    _, img_prob = predict_img(img_model, img_array, mapper)
    

lemmatizer, tokenizer, stop_words, lstm = load_txt_utils()
vgg16 = load_img_utils()
mapper, best_weights = load_model_utils()
seq = process_txt('Zazie dans le métro est un livre intéressant de Raymond Queneau', tokenizer)

print(seq)
print(type(seq))
_ , prob1 = predict_text(lstm, seq, mapper)
print(prob1)
img = path_to_img(resolve_path('data/zazie.jpg'))
img = process_img(img)
_, prob2 = predict_img(vgg16, img, mapper)
print(prob2)

agg_pred = agg_prediction(prob1, prob2, mapper, best_weights)
print(agg_pred)

#### Creeer une classe model pour faciliter 