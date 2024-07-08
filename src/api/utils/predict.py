import importlib
import os
from api.utils.resolve_path import resolve_path
from api.utils.get_models import get_model_latest_version
from dotenv import load_dotenv

def predict(models, designation, image_path):
    predictions = {}
    for i, model in enumerate(models, start=1):
        pred = model.predict(designation, image_path)
        predictions[f'{model.model_name}_{model.version}_{model.model_type}'] = pred
    return predictions

def load_models_from_file(cfg_path, model_list_file):
    models = []

    with open(model_list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        
        # Parse the line
        parts = line.split(',')
        if len(parts) != 3:
            print(f"Invalid format in model_list.txt : {line}")
            continue
        
        folder_type = parts[0].strip()
        model_name = parts[1].strip()
        version = parts[2].strip()

        # Determine is_production based on the first occurrence of 'production'
        if folder_type == 'production':
            is_production = True
        else:
            is_production = False
            
        # If version is 'latest', get the latest version
        if version == 'latest':
            version = get_model_latest_version(cfg_path, is_production, model_name)
            if version is None:
                print(f"No versions found for model {model_name} in S3 bucket")
                continue
            else:
                print(f"Latest version found for {model_name}: {version}")

        # Construct module path
        module_path = resolve_path(f'src/mlprojects/{folder_type}/{model_name}.py')

        # Load model module dynamically
        spec = importlib.util.spec_from_file_location(model_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get model class dynamically
        model_class = getattr(module, 'tf_trimodel')

        # Instantiate model and add to list
        model_instance = model_class(model_name=model_name, model_type=folder_type, version=version)
        models.append(model_instance)
    
    return models

load_dotenv(resolve_path('.env/.env.development'))
aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
mdl_list = load_models_from_file(aws_config_path,resolve_path('models/model_list.txt'))
pred = mdl_list[0].predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg'))
print(pred)
print(predict(mdl_list, 'Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg')))