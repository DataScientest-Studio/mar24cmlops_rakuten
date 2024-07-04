def predict(models, designation, image_path):
    predictions = {}
    for i, model in enumerate(models, start=1):
        pred = model.predict(designation, image_path)
        predictions[f'model_{i}'] = pred
    return predictions

def get_model_list():
    return None

model_list = get_model_list()