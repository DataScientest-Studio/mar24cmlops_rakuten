import requests

#print(requests.post(url='http://127.0.0.1:8001/prediction_text?designation=import%20jeu%20video%20japon').json())
#print(requests.post(url='http://127.0.0.1:8001/prediction_image?imageid=234234&productid=184251&directory=data/preprocessed/image_train').json())
#print(requests.post(url='http://127.0.0.1:8001/prediction?imageid=234234&productid=184251&directory=data/preprocessed/image_train&designation=import%20jeu%20video%20japon').json())
#print(requests.post(url='http://127.0.0.1:8001/prediction_image_object?',files={'image':open('data/preprocessed/image_train/image_234234_product_184251.jpg','rb')}))
#url='http://127.0.0.1:8001/image3?'
#files={'file':open('data/preprocessed/image_train/image_234234_product_184251.jpg','rb')}
#print(requests.post(url,files=files).json())
#url='http://127.0.0.1:8001/image4?'
#files={'file':open('data/preprocessed/image_train/image_234234_product_184251.jpg','rb')}
#print(requests.post(url,files=files).json())

print(requests.post(url='http://127.0.0.1:8001/predictionT?designation=import%20jeu%20video%20japon').json())
print(requests.post(url='http://127.0.0.1:8001/predictionT?imageid=234234&productid=184251&directory=data/preprocessed/image_train').json())
print(requests.post(url='http://127.0.0.1:8001/predictionT?imageid=234234&productid=184251&directory=data/preprocessed/image_train&designation=import%20jeu%20video%20japon').json())
#print(requests.post(url='http://127.0.0.1:8001/prediction_image_object?',files={'image':open('data/preprocessed/image_train/image_234234_product_184251.jpg','rb')}))
url='http://127.0.0.1:8001/predictionT?'
files={'file':open('data/preprocessed/image_train/image_234234_product_184251.jpg','rb')}
print(requests.post(url,files=files).json())

url='http://127.0.0.1:8001/predictionT?designation=import%20jeu%20video%20japon'
files={'file':open('data/preprocessed/image_train/image_234234_product_184251.jpg','rb')}
print(requests.post(url,files=files).json())