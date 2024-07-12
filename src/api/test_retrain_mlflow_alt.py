from retrain3mlflow import production_model_retrain
import json

dictionnaire=dict()
with open('param_file.json','r') as file:
    dictionnaire=json.load(file)

obj=production_model_retrain(model_type=dictionnaire.get('model_production','production'),is_production=dictionnaire.get('is_production',False))
obj.data_handle(init=dictionnaire.get('init',False),db_limitation=dictionnaire.get('db_limitation',False))
obj.train(epochs=dictionnaire.get('epochs',1))