from api.utils.tf_trimodel_travail import tf_trimodel
from api.utils.resolve_path import resolve_path
import pandas as pd
from sklearn.metrics import accuracy_score

from api.utils.make_db import process_listing
from dotenv import load_dotenv
import os
from datetime import datetime,timedelta



#df=pd.DataFrame({'description':['film fantastique','tooneinstein'],
#                'designation':['la momie','jeu video playstation cartouche'],
#                'image':[resolve_path('data/preprocessed/image_train/image_550506_product_929938.jpg'),
#                         resolve_path('data/preprocessed/image_train/image_234234_product_184251.jpg')]})

#print(df.head())
#print(df.info())

#model = tf_trimodel(model_type='production', version='latest')
#print('debut')
#print(model.batch_predict(df))
#print('repere')

if __name__=="__main__":
    global aws_config_path, duckdb_path, encrypted_file_path, conn, mdl_list, s3_conn

    # Load environment variables from .env file
    env_path = resolve_path(".envp/.env.development")
    load_dotenv(env_path)

    # Convert paths to absolute paths
    aws_config_path = resolve_path(os.environ["AWS_CONFIG_PATH"])
    duckdb_path = os.path.join(
        resolve_path(os.environ["DATA_PATH"]), os.environ["RAKUTEN_DB_NAME"].lstrip("/")
    )
    encrypted_file_path = os.path.join(
        resolve_path(os.environ["AWS_CONFIG_FOLDER"]), ".encrypted"
    )
    
    X_pathway=resolve_path(os.path.join('data','X_train.csv'))
    Y_pathway=resolve_path(os.path.join('data','Y_train.csv'))
    df=process_listing(X_pathway,Y_pathway)
    print(df.head())
    longueur=df.shape[0]
    init_ind=int(longueur*3/10)
    init_date=datetime.strptime("2024-07-01","%Y-%m-%d")
    df.loc[0:init_ind,'validate_datetime']=init_date
    print(init_date)
    for i in range(init_ind,(longueur-init_ind),100):
        df.loc[(init_ind+i):(init_ind+i+99),'validate_datetime']=init_date+timedelta(days=int(i/100))
    ### Prediction sur les 30% 
    dfr=df[df['validate_datetime']==init_date]
    dfr['image']=os.path.join(resolve_path('data/preprocessed/image_train'),'image_')+dfr['imageid'].astype(str)+'_product_'+dfr['productid'].astype(str)+'.jpg'
    print(dfr[['description','designation','image']])
    print(dfr.info())
    #print(dfr['image'].tolist())
    model = tf_trimodel(model_type='production', version='latest')
    print('debut')
    #print(model.batch_predict(dfr[['description','designation','image']]))
    print('repere')