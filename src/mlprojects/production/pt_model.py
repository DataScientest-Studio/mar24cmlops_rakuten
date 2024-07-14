import os
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from api.utils.resolve_path import resolve_path

# Chargement des données

base_path = resolve_path('data/test_model/')

X_train = pd.read_csv(resolve_path("data/X_train.csv"), index_col=0)
Y_train = pd.read_csv(resolve_path("data/Y_train.csv"), index_col=0)
listing_df = X_train.join(Y_train)

num_listings = 5000  # Réduire la taille du jeu de données
num_epochs = 15
batch_size = 32  # Réduire la taille du lot

listing_df = listing_df.sample(num_listings)

# Prétraitement des textes
texts = (listing_df["description"] + " " + listing_df["designation"]).astype(str)
listing_df["image_path"] = listing_df.apply(
    lambda row: resolve_path(f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"),
    axis=1,
)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Sauvegarde du tokenizer
with open(os.path.join(base_path,'tokenizer.pkl'), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Charger et prétraiter les images
def load_and_preprocess_image(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

image_paths = listing_df['image_path'].values
images = np.array([load_and_preprocess_image(path) for path in image_paths])

# Encoder les labels
modalite_mapping = {int(modalite): i for i, modalite in enumerate(listing_df['prdtypecode'].unique())}
listing_df['target_prdtypecode'] = listing_df['prdtypecode'].replace(modalite_mapping)
labels = listing_df['target_prdtypecode']
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes=num_classes)

# Sauvegarde du mapping des modalités
# modalite_mapping_str_keys = {str(key): value for key, value in modalite_mapping.items()}
# with open(os.path.join(base_path,'modalite_mapping.json'), 'w') as json_file:
#     json.dump(modalite_mapping_str_keys, json_file)
with open(os.path.join(base_path,'modalite_mapping.pkl'), 'wb') as pickle_file:
    pickle.dump(modalite_mapping, pickle_file)

# Diviser les données en ensembles d'entraînement et de test
X_train_texts, X_test_texts, X_train_images, X_test_images, y_train, y_test = train_test_split(
    padded_sequences, images, labels, test_size=0.2, random_state=42
)

# Modèle LSTM pour les textes
def create_text_model():
    text_input = tf.keras.Input(shape=(100,), name='text_input')
    x = tf.keras.layers.Embedding(input_dim=10000, output_dim=64)(text_input)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    text_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='text_output')(x)
    return tf.keras.Model(inputs=text_input, outputs=text_output)

text_model = create_text_model()

# Compiler et entraîner le modèle textuel seul
text_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

text_model.fit(
    X_train_texts, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(X_test_texts, y_test)
)

# Sauvegarder le modèle textuel
text_model.save(os.path.join(base_path,"text_model.keras"))

# Modèle VGG16 pour les images
def create_image_model():
    image_input = tf.keras.Input(shape=(224, 224, 3), name='image_input')
    base_model = tf.keras.applications.VGG16(include_top=False, input_tensor=image_input)
    base_model.trainable = False  # Fine-tuning
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    image_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='image_output')(x)
    return tf.keras.Model(inputs=image_input, outputs=image_output)

image_model = create_image_model()

# Compiler et entraîner le modèle d'image seul
image_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

image_model.fit(
    X_train_images, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(X_test_images, y_test)
)

# Sauvegarder le modèle d'image
image_model.save(os.path.join(base_path,"image_model.keras"))

# Désactiver l'entraînement pour les modèles préentraînés
text_model.trainable = False
image_model.trainable = False

# Modèle combiné
def create_combined_model():
    combined_input = tf.keras.layers.concatenate([text_model.output, image_model.output])
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='combined_output')(combined_input)
    return tf.keras.Model(inputs=[text_model.input, image_model.input], outputs=x)

combined_model = create_combined_model()

# Compiler le modèle combiné
combined_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner le modèle combiné
combined_model.fit(
    [X_train_texts, X_train_images], y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=([X_test_texts, X_test_images], y_test)
)

# Sauvegarder le modèle combiné
combined_model.save(os.path.join(base_path,"combined_model.keras"))