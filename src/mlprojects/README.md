README
==============================

This directory aims at keeping all mlprojects files for ML reproducibilty.
Each subfolder is a machine learning model.
A data scientist must provide the mlproject files and three functions (for the /predict_typecode and /model_retrain enpoints):
* load_model: a function to load his model
* preprocess: a preprocessing function
* predict: a prediction function
* train: a train function (in case the model needs to be retrained)