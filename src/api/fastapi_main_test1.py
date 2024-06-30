from fastapi import FastAPI, HTTPException, Depends, status
from prediction import predict_with_unified_interface
from fastapi import UploadFile, File
import os
import uvicorn
from dotenv import load_dotenv
import duckdb
from utils.make_db import download_initial_db
from utils.security import create_access_token, verify_password
from utils.s3_utils import create_s3_conn_from_creds, download_from_s3
import boto3 
import pandas as pd 

s3_client=None
db_conn=None
prd_categories=dict()

# Load environment variables from .env file
load_dotenv('.env/.env.development')
    
aws_config_path = os.environ['AWS_CONFIG_PATH']
duckdb_path = os.path.join(os.environ['DATA_PATH'], os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']
# Download database for the mapping of the results    
db_conn = duckdb.connect(database=duckdb_path, read_only=False)
s3_client = create_s3_conn_from_creds(os.getenv('AWS_CONFIG_PATH'))

# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def get_index():
    return {'data': 'hello world '+os.getcwd()}

@app.post('/predict')
#async def prediction(designation : str =None, imageid : int = None, productid : int = None, directory : str = 'image_train', new_image : str = None, file : UploadFile | None = None):
async def prediction(designation : str =None, imageid : int = None, productid : int = None, directory : str = 'image_train', new_image : str = None, file : UploadFile = File(None)):

    img_context=None
    if file is not None :
        img_context= await file.read()
    return {'prediction' : predict_with_unified_interface(s3_client=s3_client, designation=designation, imageid=imageid,productid=productid,directory=directory,new_image=new_image,file=img_context)}



if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)
