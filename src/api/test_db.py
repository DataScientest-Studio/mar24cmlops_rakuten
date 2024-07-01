from dotenv import load_dotenv
import duckdb
import os
from utils.make_db import download_initial_db
from utils.security import create_access_token, verify_password
from utils.s3_utils import create_s3_conn_from_creds, download_from_s3
import boto3 
from io import BytesIO
import pandas as pd
import numpy as np
 
# Load environment variables from .env file
load_dotenv('.env/.env.development')
    
aws_config_path = os.environ['AWS_CONFIG_PATH']
duckdb_path = os.path.join(os.environ['DATA_PATH'], os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']
    
# Check if the DuckDB database file exists locally, if not, download it from S3
if not os.path.isfile(duckdb_path):
    print('No Database Found locally')
    # Since no database found for the API, download the initial database from S3
    download_initial_db(aws_config_path, duckdb_path)
    print('Database Sucessfully Downloaded')
        
# Load DuckDB connection   
conn = duckdb.connect(database=duckdb_path, read_only=False)
print(os.getcwd())

#s3_client = create_s3_conn_from_creds(os.path.join(os.getenv('AWS_CONFIG_PATH'),'.aws_config_decr.ini'))
s3_client = create_s3_conn_from_creds(os.getenv('AWS_CONFIG_PATH'))
#s3_client.download_file("rakutenprojectbucket",'image_train/image_234234_product_184251.jpg', os.path.join(os.getcwd(),'coucou.jpg'))
#download_from_s3(s3_conn=s3_client, 
#                 bucket_path ='X_test.csv',
#                 local_path =os.path.join(os.getcwd(),'coucou.csv'))
download_from_s3(s3_conn=s3_client, 
                 bucket_path ='image_train/image_1007738129_product_435130019.jpg',
                 local_path =os.path.join(os.getcwd(),'coucou.jpg'))

liste=[i['Key'] for i in s3_client.list_objects(Bucket="rakutenprojectbucket", Prefix="image_train")['Contents']]
if 'image_train/image_234234_product_184251.jpg' in liste:
    print('ouaips')
print(len(liste))
print(conn.sql('select * from fact_listings where imageid=234234;').fetchone())
print(conn.sql('select listing_id from fact_listings where imageid=234234;').fetchone())
print(conn.sql('show table fact_listings;').fetchall())
liste2=[i[0] for i in conn.sql('show table fact_listings;').fetchall()]
for i in liste2:
    a=conn.sql(f'select {i} from fact_listings where imageid=234234;').fetchone()
    print(f'{i} : {a[0]}')
    
#print(conn.sql('select * from dim_prdtypecode;').fetchall())

print(conn.sql('select * from fact_listings limit 10;').fetchall())


results = conn.sql("select * from fact_listings limit 10;").df()
print(results)
print(results.info())

#listeA=conn.sql('select imageid,user_prdtypecode from fact_listings;').fetchall()
#print(listeA)
#for i in np.random.choice(np.array(len(listeA)),100):
#    download_from_s3(s3_conn=s3_client, 
#                 bucket_path =f"image_train/image_{listeA[i][0]}_product_{listeA[i][1]}.jpg",
#                 local_path =os.path.join(os.getcwd(),'coucou.jpg'))
    
#liste=os.listdir('/mnt/c/Users/USER/preprocessed/image_test/')
#print(liste)
liste=os.listdir('/mnt/c/Users/USER/Projet-Rakuten/images/image_test/')
#print(liste)
#print(liste[10][:-4])
listeA=[i for i in liste if i[-4:]=='.jpg']
print(listeA)
for i in np.random.choice(np.array(len(listeA)),100):
    download_from_s3(s3_conn=s3_client, 
                 bucket_path =f"image_test/{listeA[i]}",
                local_path =os.path.join(os.getcwd(),'coucou.jpg'))