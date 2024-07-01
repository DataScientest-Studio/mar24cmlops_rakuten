import os
from api.utils.s3_utils import create_s3_conn_from_creds, download_from_s3, upload_folder_to_s3
from api.utils.resolve_path import resolve_path
from dotenv import load_dotenv
from datetime import datetime

def download_production_model(cfg_path, folder_name, local_download_path):
    """
    Download a specific folder from the production model repository in S3.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        folder_name (str): The name of the folder to download from the S3 bucket.
        local_download_path (str): The local path where the folder will be downloaded.
    """
    # Create S3 connection
    s3_conn = create_s3_conn_from_creds(cfg_path)

    # Define S3 bucket and prefix
    bucket_name = "rakutenprojectbucket"
    s3_prefix = f"model_repository/production/{folder_name}/"

    # List objects in the specified folder
    try:
        response = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
        if 'Contents' not in response:
            print(f"No objects found in S3 bucket at {s3_prefix}")
            return None
        
        # Create local directory if it doesn't exist
        os.makedirs(local_download_path, exist_ok=True)

        # Download each file in the folder
        for obj in response['Contents']:
            s3_key = obj['Key']
            local_file_path = os.path.join(local_download_path, os.path.relpath(s3_key, s3_prefix))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            download_from_s3(s3_conn, s3_key, local_file_path)
            print(f"Downloaded {s3_key} to {local_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def download_init_prod_model(cfg_path, local_download_path):
    """
    Download the initial model from the production model repository in S3.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        local_download_path (str): The local path where the folder will be downloaded.
    """
    download_production_model(cfg_path,'init_model', local_download_path)
    return None

def list_model_repository_folders(cfg_path, is_production):
    """
    List all folders in the model_repository directory of the S3 bucket.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        is_production (bool): if not production then staging

    Returns:
        list: A list of folder names in the model_repository directory.
    """
    folder_type = 'production' if is_production else 'staging'
    
    # Create S3 connection
    s3_conn = create_s3_conn_from_creds(cfg_path)

    # Define S3 bucket and prefix
    bucket_name = "rakutenprojectbucket"
    s3_prefix = f"model_repository/{folder_type}/"

    # List objects in the specified prefix
    try:
        response = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix, Delimiter='/')
        if 'CommonPrefixes' not in response:
            print(f"No folders found in S3 bucket at {s3_prefix}")
            return []
        
        folders = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
        return folders

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def upload_model_to_repository(cfg_path, local_model_folder, is_production):
    """
    Upload a model folder to the S3 model repository with datetime-based folder name.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        local_model_folder (str): The local model folder to be uploaded.

    Returns:
        str: The name of the folder in S3 where the files were uploaded.
    """
    folder_type = 'production' if is_production else 'staging'
    current_datetime = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    s3_folder_prefix = f"model_repository/{folder_type}/"
    s3_folder = os.path.join(s3_folder_prefix, current_datetime).replace("\\", "/") + "/"

    upload_folder_to_s3(cfg_path, local_model_folder, s3_folder)
    
    return None

# Load environment variables from .env file
# env_path = resolve_path('.env/.env.development')
# load_dotenv(env_path)
# aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])

# upload_model_to_repository(aws_config_path, resolve_path('models/production_model/'), False)
# list_models = list_model_repository_folders(aws_config_path)
# download_production_model(aws_config_path, '20240628_17-56-24' ,resolve_path('models/production_model/'))