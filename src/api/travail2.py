from api.utils.tf_trimodel_travail import tf_trimodel
from api.utils.resolve_path import resolve_path
import pandas as pd
from sklearn.metrics import accuracy_score
from api.utils.make_db import process_listing
from dotenv import load_dotenv
import os
from datetime import datetime,timedelta
import duckdb
import argparse


def img_path_create(df):
    df['image']=os.path.join(resolve_path('data/preprocessed/image_train'),'image_')+df['imageid'].astype(str)+'_product_'+df['productid'].astype(str)+'.jpg'
    df.drop(['imageid','productid'],axis=1,inplace=True)
    
def df_create():
    # Load DuckDB connection
    db_conn = duckdb.connect(database=duckdb_path, read_only=False)
    df=db_conn.sql('SELECT designation,description,imageid,productid,user_prdtypecode,validate_datetime FROM fact_listings;').df()
    img_path_create(df)
    df=df.dropna(subset=['user_prdtypecode'],how='any')    
    df['user_prdtypecode']=df['user_prdtypecode'].astype(int)
    return df


def date_initialization(df):
    longueur=df.shape[0]
    init_ind=int(longueur*3/10) # 30%
    df.loc[0:init_ind,'validate_datetime']=init_date
        
    # modification du dataframe avec un gradient de date
    for i in range(0,(longueur-init_ind),100):
        df.loc[(init_ind+i):(init_ind+i+99),'validate_datetime']=init_date+timedelta(days=int(i/100))

def db_restriction(df,i_date,f_date):
    if f_date is None:
        return df[(df['validate_datetime']==i_date)]
    else :
        return df[(df['validate_datetime']>=i_date)&(df['validate_datetime']<=f_date)]

def prediction(df,model):
    return model.batch_predict(dfr[['description','designation','image']])

#if __name__=='__main__':
def execution(model_type='production',model_name='production_model_retrain',production='staging',version='lastest',i_date=None,f_date=None)
    global aws_config_path, duckdb_path, encrypted_file_path, conn, mdl_list, s3_conn, init_date
    init_date=datetime.strptime("2024-07-01","%Y-%m-%d")

    
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

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--i_date", help="start date",default=init_date)
    #parser.add_argument("--f_date",help="end date",default=None)
    #parser.add_argument('--version',default='latest')
    #parser.add_argument('--production',default='staging')
    #parser.add_argument('--model_name',default='production_model_retrain')
    #args = parser.parse_args()

    #model = tf_trimodel(model_type=args.production, version=args.version) # un seul à déclarer, ne pas les multiplier
    model = tf_trimodel(model_type=model_type, model_name=model_name,version=version) # un seul à déclarer, ne pas les multiplier
    
    df=df_create()
    date_initialization(df)            
    # Prédiction
    if i_date is None:
        i_date=init_date
    if f_date is None:
        f_date=i_date+timedelta(days=1)
    dfr=db_restriction(df,i_date,f_date)
    result=prediction(dfr,model)
    with open('estimation.txt','a') as file:
        file.write(f"{i_date} : {accuracy_score(result,dfr['user_prdtypecode'])}\n")

