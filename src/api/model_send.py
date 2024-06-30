from utils.make_db import download_initial_db
from utils.security import create_access_token, verify_password
from utils.s3_utils import create_s3_conn_from_creds, download_from_s3, upload_to_s3
import boto3 
from dotenv import load_dotenv
import os
from glob import glob
from datetime import datetime

# Load environment variables from .env file
load_dotenv('.env/.env.development')
    
aws_config_path = os.environ['AWS_CONFIG_PATH']
duckdb_path = os.path.join(os.environ['DATA_PATH'], os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']
# Download database for the mapping of the results    
s3_client = create_s3_conn_from_creds(os.getenv('AWS_CONFIG_PATH'))

liste1=os.listdir('models/production_model')
liste2=[i for i in liste1 if (i[-4:]=='.pkl' or i[-6:]=='.keras')]
print(liste2)
jour=datetime.now().strftime("%Y%m%d_%H-%M-%S")

for i in liste2:
    print(f"model_repository/production/{jour}/{i}")
    upload_to_s3(s3_client,os.path.join("models/production_model",i),f"model_repository/production/{jour}/{i}")