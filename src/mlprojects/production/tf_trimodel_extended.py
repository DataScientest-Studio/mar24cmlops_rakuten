from mlprojects.production.tf_trimodel import tf_trimodel
import numpy as np 
import os 
import pickle
from scipy.optimize import minimize
from api.utils.resolve_path import resolve_path
from api.utils.make_db import process_listing

class tf_trimodel_extended(tf_trimodel):
    """
    Extended class of tf_trimodel with additional methods to retrain the model 
    and the aggregation layer.
    """

    def __init__(self, model_name, version, model_type):
        """
        Initialize the tf_trimodel_extended class.
        
        Args:
            model_name (str): Name of the model.
            version (str): Version of the model.
            model_type (str): Type of the model (production or staging).
        """
        super().__init__(model_name, version, model_type)
        
    def train_model(self, new_text_data, new_image_data, new_labels, epochs=10, batch_size=32):
        """
        Retrain the models with new data.
        
        Args:
            new_text_data (list of str): New text data for retraining.
            new_image_data (list of str or bytes): New image data for retraining.
            new_labels (list of int): New labels for retraining.
            epochs (int): Number of epochs for retraining.
            batch_size (int): Batch size for retraining.
        """
        # Update tokenizer with new text data
        self.tokenizer.fit_on_texts(new_text_data)
        
        # Save the updated tokenizer
        tokenizer_json = self.tokenizer.to_json()
        model_path = self.get_model_path()
        with open(os.path.join(model_path, "tokenizer_config.json"), "w", encoding="utf-8") as json_file:
            json_file.write(tokenizer_json)
        
        # Process the new text data
        processed_texts = [self.process_txt(text) for text in new_text_data]
        text_sequences = np.array([sequence[0] for sequence in processed_texts])
        
        # Process the new image data
        processed_images = [self.process_img(self.path_to_img(img)) if isinstance(img, str) else self.process_img(self.byte_to_img(img)) for img in new_image_data]
        image_arrays = np.array(processed_images)
        
        # Transform the labels
        new_labels = np.array(new_labels)
        
        # Compile the LSTM text model
        self.lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Retrain the LSTM text model
        self.lstm.fit(text_sequences, new_labels, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Compile the VGG16 image model
        self.vgg16.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Retrain the VGG16 image model
        self.vgg16.fit(image_arrays, new_labels, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save the retrained models
        self.lstm.save(os.path.join(model_path, "retrained_lstm_model.keras"))
        self.vgg16.save(os.path.join(model_path, "retrained_vgg16_model.keras"))

    def retrain_agg_layer(self, new_text_data, new_image_data, new_labels, epochs=10, batch_size=32):
        """
        Retrain the aggregation layer with new data.
        
        Args:
            new_text_data (list of str): New text data for retraining.
            new_image_data (list of str or bytes): New image data for retraining.
            new_labels (list of int): New labels for retraining.
            epochs (int): Number of epochs for retraining.
            batch_size (int): Batch size for retraining.
        """
        # Process the new text data
        processed_texts = [self.process_txt(text) for text in new_text_data]
        text_sequences = np.array([sequence[0] for sequence in processed_texts])
        
        # Process the new image data
        processed_images = [self.process_img(self.path_to_img(img)) if isinstance(img, str) else self.process_img(self.byte_to_img(img)) for img in new_image_data]
        image_arrays = np.array(processed_images)
        
        # Transform the labels
        new_labels = np.array(new_labels)

        # Get the probabilities from the text and image models
        text_probs = np.array([self.predict_text(seq)[1] for seq in text_sequences])
        image_probs = np.array([self.predict_img(img)[1] for img in image_arrays])

        # Define the objective function for optimization
        def objective(weights):
            weighted_probs = weights[0] * text_probs + weights[1] * image_probs
            predictions = np.argmax(weighted_probs, axis=1)
            accuracy = np.mean(predictions == new_labels)
            return -accuracy  # Minimize negative accuracy

        # Initial weights
        initial_weights = self.best_weights
        
        # Optimize the weights
        result = minimize(objective, initial_weights, method='Nelder-Mead')
        optimized_weights = result.x

        # Update the best weights
        self.best_weights = optimized_weights
        
        # Save the updated best weights
        model_path = self.get_model_path()
        with open(os.path.join(model_path, "best_weights.pkl"), "wb") as file:
            pickle.dump(self.best_weights, file)
        
# Exemple d'utilisation de la classe étendue
#tf_model_extended = tf_trimodel_extended(model_name='tf_trimodel', version='20240708_18-15-54', model_type='production')

listing_df = process_listing(resolve_path('data/X_train.csv'),resolve_path('data/Y_train.csv'))
listing_df = listing_df.head(1000)

new_text_data = list(listing_df['designation'])
new_image_data = listing_df.apply(lambda row: resolve_path(f"image_{row['imageid']}_product_{row['productid']}"), axis=1).tolist()
new_labels = list(listing_df['user_prdtypecode'])

tf_model_extended.retrain_model(new_text_data, new_image_data, new_labels)
tf_model_extended.retrain_agg_layer(new_text_data, new_image_data, new_labels)