from api.utils.tf_trimodel_travail import tf_trimodel
from api.utils.resolve_path import resolve_path
import pandas as pd
from sklearn.metrics import accuracy_score

from api.utils.make_db import process_listing
from dotenv import load_dotenv
import os
from datetime import datetime,timedelta

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
    #init_ind=int(longueur*3/10)
    init_ind=int(longueur/500) ####### en attendant
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
    model = tf_trimodel(model_type='production', version='latest')
    print('debut')
    result=model.batch_predict(dfr[['description','designation','image']])
    print(result)
    print('repere')
    print(accuracy_score(result,dfr['user_prdtypecode']))
    print(df['validate_datetime'].unique())
    #print(sort(df['validate_datetime'].unique()))
    print(df['validate_datetime'][:100])
    for i_date in sorted(df['validate_datetime'].unique()):
        dfri=df[df['validate_datetime']==i_date]
        dfri['image']=os.path.join(resolve_path('data/preprocessed/image_train'),'image_')+dfri['imageid'].astype(str)+'_product_'+dfri['productid'].astype(str)+'.jpg'
        #print(dfr[['description','designation','image']])
        #print(dfr.info())
        model = tf_trimodel(model_type='production', version='latest')
        #print('debut')
        result=model.batch_predict(dfri[['description','designation','image']])
        #print(result)
        #print('repere')
        with open('estimation.txt','a') as file:
            file.write(f"{i_date} : {accuracy_score(result,dfri['user_prdtypecode'])}\n")
        print(i_date)
        print(accuracy_score(result,dfri['user_prdtypecode']))
    