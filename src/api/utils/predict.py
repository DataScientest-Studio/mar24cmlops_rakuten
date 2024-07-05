import importlib
import os
from api.utils.resolve_path import resolve_path

def predict(models, designation, image_path):
    predictions = {}
    for i, model in enumerate(models, start=1):
        pred = model.predict(designation, image_path)
        predictions[f'model_{i}'] = pred
    return predictions

def load_models_from_file(file_path):
    models = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                model_info = line.split(',')
                if len(model_info) == 3:
                    model_type = model_info[0].strip()
                    model_name = model_info[1].strip()
                    model_datetime = model_info[2].strip()

                    # Construction du chemin vers le module du modèle
                    module_path = f'mlprojects/{model_type}/{model_name}.py'

                    # Charger le module du modèle
                    spec = importlib.util.spec_from_file_location(model_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Récupérer la classe du modèle (supposant que le nom de la classe est Model)
                    model_class = getattr(module, f'{model_name}')

                    # Instancier le modèle et l'ajouter à la liste
                    model_instance = model_class()
                    models.append(model_instance)
    return models

mdl_list = load_models_from_file(resolve_path('models/model_list.txt'))
pred = mdl_list[0].predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg'))
print(pred)