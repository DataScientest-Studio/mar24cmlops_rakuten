from api.utils.tf_trimodel_travail import tf_trimodel
from api.utils.resolve_path import resolve_path
import pandas as pd

df=pd.DataFrame({'description':['film fantastique','tooneinstein'],
                'designation':['la momie','jeu video playstation cartouche'],
                'image':[resolve_path('data/preprocessed/image_train/image_550506_product_929938.jpg'),
                         resolve_path('data/preprocessed/image_train/image_234234_product_184251.jpg')]})

print(df.head())
print(df.info())

model = tf_trimodel(model_type='production', version='latest')
print('debut')
print(model.batch_predict(df))
print('repere')

#prediction = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg'))
#print(df.loc[0,'designation'])
#print(df.loc[0,'image'])
#prediction = model.predict(df.loc[0,'designation'],df.loc[0,'image'])
#print(prediction)
#prediction = model.predict(df.loc[1,'designation'],df.loc[1,'image'])
#print(prediction)
#print(df['designation'])
#print(df['image'])
#predictions = model.batch_predict(df['designation'],df['image'])
#print(predictions)
# prediction_from_bytes = model.predict('Zazie dans le métro est un livre intéressant de Raymond Queneau', image_bytes)
# print(prediction_from_bytes)