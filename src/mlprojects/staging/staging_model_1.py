import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from torchvision import transforms
import os
from api.utils.resolve_path import resolve_path
import re
import numpy as np
import json
from io import BytesIO
from PIL import Image
import pickle

class staging_model_1:
    def __init__(self, model_type='production', version='latest', model_name=None):
        self.model_type = model_type
        self.version = version
        self.model_name = model_name
        self.lemmatizer = None
        self.tokenizer = None
        self.stop_words = None
        self.lstm = None
        self.vgg16 = None
        self.mapper = None
        self.best_weights = None
        self.load_txt_utils()
        self.load_img_utils()
        self.load_model_utils()
        
    def get_model_path(self):
        if self.model_type == 'production':
            folder = 'production_model'
        else:
            folder = 'staging_models'
            if self.version == 'latest':
                base_path = resolve_path(f'models/{folder}/')
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
        with open(os.path.join(model_path, "tokenizer_config.json"), "r", encoding="utf-8") as json_file:
            tokenizer_config = json_file.read()
        self.tokenizer = tokenizer_from_json(tokenizer_config)
        self.lstm = torch.load(os.path.join(model_path, "best_lstm_model.pth"))
    
    def load_img_utils(self):
        model_path = self.get_model_path()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Linear(4096, len(self.mapper))  # Adjust the output layer to match the number of classes
        self.vgg16.load_state_dict(torch.load(os.path.join(model_path, "best_vgg16_model.pth")))
    
    def load_model_utils(self):
        model_path = self.get_model_path()
        with open(os.path.join(model_path, "mapper.json"), "r") as json_file:
            self.mapper = json.load(json_file)
        with open(os.path.join(model_path, "best_weights.pkl"), "rb") as file:
            self.best_weights = pickle.load(file)
    
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
            return torch.tensor(padded_sequences, dtype=torch.long)
        return None
    
    def path_to_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def byte_to_img(self, file):
        img = Image.open(BytesIO(file)).convert('RGB')
        return img

    def process_img(self, img):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = preprocess(img)
        return img_tensor
    
    def predict_text(self, text_sequence):
        self.lstm.eval()
        with torch.no_grad():
            output = self.lstm(text_sequence)
            probability = torch.softmax(output, dim=1).numpy()
            pred = np.argmax(probability)
            return int(self.mapper[str(pred)]), probability
    
    def predict_img(self, img_tensor):
        self.vgg16.eval()
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.vgg16(img_tensor)
            probability = torch.softmax(output, dim=1).numpy()
            pred = np.argmax(probability)
            return int(self.mapper[str(pred)]), probability
    
    def agg_prediction(self, txt_prob, img_prob):
        concatenate_proba = (self.best_weights[0] * txt_prob + self.best_weights[1] * img_prob)
        pred = np.argmax(concatenate_proba)
        return int(self.mapper[str(pred)])
    
    def predict(self, designation, image_path):
        text_sequence = self.process_txt(designation)
        img = self.path_to_img(image_path)
        img_tensor = self.process_img(img)
        
        _, txt_prob = self.predict_text(text_sequence)
        _, img_prob = self.predict_img(img_tensor)
        
        agg_pred = self.agg_prediction(txt_prob, img_prob)
        return agg_pred
    
    def train(self, train_data, val_data, epochs=10, lr=0.001, model_save_path=None):
        criterion = nn.CrossEntropyLoss()
        lstm_optimizer = optim.Adam(self.lstm.parameters(), lr=lr)
        vgg16_optimizer = optim.Adam(self.vgg16.parameters(), lr=lr)

        for epoch in range(epochs):
            self.lstm.train()
            self.vgg16.train()
            running_loss = 0.0
            for texts, imgs, labels in train_data:
                text_sequences = [self.process_txt(text) for text in texts]
                img_tensors = torch.stack([self.process_img(img) for img in imgs])
                labels = torch.tensor(labels, dtype=torch.long)
                
                # Zero the parameter gradients
                lstm_optimizer.zero_grad()
                vgg16_optimizer.zero_grad()

                # Forward pass
                text_outputs = self.lstm(text_sequences)
                img_outputs = self.vgg16(img_tensors)

                loss = criterion(text_outputs, labels) + criterion(img_outputs, labels)
                loss.backward()
                lstm_optimizer.step()
                vgg16_optimizer.step()

                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_data)}")

        if model_save_path:
            torch.save(self.lstm.state_dict(), os.path.join(model_save_path, "best_lstm_model.pth"))
            torch.save(self.vgg16.state_dict(), os.path.join(model_save_path, "best_vgg16_model.pth"))
            print(f"Models saved to {model_save_path}")

# Utilisation de la classe Model
# model = Model(model_type='staging', version='latest')
# train_data = ... # Define your training data
# val_data = ... # Define your validation data
# model.train(train_data, val_data, epochs=10, lr=0.001, model_save_path=resolve_path('models/staging/latest'))
