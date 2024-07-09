import os
from api.utils.s3_utils import create_s3_conn_from_creds, download_from_s3, upload_folder_to_s3
from api.utils.resolve_path import resolve_path
from dotenv import load_dotenv
from datetime import datetime

def download_model(cfg_path, model_name, version, local_download_path, is_production):
    """
    Download a specific model version from the model repository in S3.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        model_name (str): The name of the model to download from the S3 bucket.
        version (str): The version of the model to download (format: 'YYYYMMDD_HH-MM-SS' or 'latest').
        local_download_path (str): The local path where the model will be downloaded.
        is_production (bool): If True, download from the production directory; otherwise, download from the staging directory.
        
    Example:
        load_dotenv(resolve_path('.env/.env.development'))
        aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
        download_model(aws_config_path, 'tf_trimodel', 'latest', resolve_path('data/test_dl_model/'), is_production = True)
    """
    # Create S3 connection
    s3_conn = create_s3_conn_from_creds(cfg_path)

    # Determine folder type based on is_production
    folder_type = 'production' if is_production else 'staging'

    # Define S3 bucket and prefix
    bucket_name = "rakutenprojectbucket"
    
    # Check if we are downloading the initial model
    if model_name == 'init_model':
        s3_prefix = f"model_repository/{folder_type}/init_model/"
    else:
        # If version is 'latest', determine the latest version
        if version == 'latest':
            version = get_model_latest_version(cfg_path, is_production, model_name)
            if version is None:
                print(f"No versions found for model {model_name} in S3 bucket")
                return None
            else:
                print(f"Latest version found for {model_name}: {version}")
        
        # Define S3 prefix with specific version
        s3_prefix = f"model_repository/{folder_type}/{model_name}/{version}/"

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
        
def verify_and_download_models(cfg_path, model_list_file):
    """
    Verify the presence of models specified in model_list_file and download if missing.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        model_list_file (str): Path to the model list file (model_list.txt).
    """
    with open(model_list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        
        # Parse the line
        parts = line.split(',')
        if len(parts) != 3:
            print(f"Invalid format in model_list.txt : {line}")
            continue
        
        folder_type = parts[0].strip()
        model_name = parts[1].strip()
        version = parts[2].strip()

        # Determine is_production based on the first occurrence of 'production'
        if folder_type == 'production':
            is_production = True
            folder_type = 'production_model'  # Use specific folder type for production
        else:
            is_production = False
            folder_type = 'staging_models'

        # If version is 'latest', get the latest version from S3
        if version == 'latest':
            version = get_model_latest_version(cfg_path, is_production, model_name)
            if version is None:
                print(f"No versions found for model {model_name} in S3 bucket")
                continue
            else:
                print(f"Latest version found for {model_name}: {version}")

        # Define local download path based on model_name and version
        local_download_path = os.path.join('models/', folder_type, model_name, version)
        local_download_path = resolve_path(local_download_path)

        # Check if the model already exists
        if not os.path.exists(local_download_path):
            print(f"Model {model_name} version {version} not found locally. Downloading...")
            download_model(cfg_path, model_name, version, local_download_path, is_production)
        else:
            print(f"Model {model_name} version {version} already exists locally.")
        
def list_model_repository_folders(cfg_path, is_production):
    """
    List all model folders and their version folders in the model_repository directory of the S3 bucket.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        is_production (bool): If True, list folders in the production directory; otherwise, list in the staging directory.

    Returns:
        dict: A dictionary where keys are model names and values are lists of version folders.
        
    Example:
        load_dotenv(resolve_path('.env/.env.development'))
        aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
        list_models = list_model_repository_folders(aws_config_path, is_production=True)
    """
    folder_type = 'production' if is_production else 'staging'
    
    # Create S3 connection
    s3_conn = create_s3_conn_from_creds(cfg_path)

    # Define S3 bucket and prefix
    bucket_name = "rakutenprojectbucket"
    s3_prefix = f"model_repository/{folder_type}/"

    # List model folders in the specified prefix
    try:
        response = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix, Delimiter='/')
        if 'CommonPrefixes' not in response:
            print(f"No folders found in S3 bucket at {s3_prefix}")
            return {}
        
        model_folders = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
        model_versions = {}

        # List version folders for each model
        for model in model_folders:
            model_prefix = f"{s3_prefix}{model}/"
            response = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=model_prefix, Delimiter='/')
            if 'CommonPrefixes' in response:
                version_folders = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
                model_versions[model] = version_folders
            else:
                model_versions[model] = []

        return model_versions

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    
def get_model_latest_version(cfg_path, is_production, model_name):
    """
    Get the latest version of a model from the model repository in S3.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        is_production (bool): If True, retrieve from the production directory; otherwise, retrieve from the staging directory.
        model_name (str): The name of the model to get the latest version for.

    Returns:
        str: The latest version of the model in format 'YYYYMMDD_HH-MM-SS', or None if no versions are found.
    
    Example:
        load_dotenv(resolve_path('.env/.env.development'))
        aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
        get_model_latest_version(aws_config_path, False, 'tf_trimodel')
    """
    # Create S3 connection
    s3_conn = create_s3_conn_from_creds(cfg_path)

    # Determine folder type based on is_production
    folder_type = 'production' if is_production else 'staging'

    # Define S3 bucket and prefix
    bucket_name = "rakutenprojectbucket"
    s3_prefix = f"model_repository/{folder_type}/{model_name}/"

    # List objects in the specified prefix
    try:
        response = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
        if 'Contents' not in response:
            print(f"No objects found in S3 bucket at {s3_prefix}")
            return None
        
        # Extract version timestamps from keys
        versions = []
        for obj in response['Contents']:
            try:
                version_str = obj['Key'].split('/')[-2]
                version_dt = datetime.strptime(version_str, '%Y%m%d_%H-%M-%S')
                versions.append((version_dt, version_str))
            except ValueError:
                continue
        
        # Find the latest version
        if versions:
            latest_version = max(versions, key=lambda x: x[0])
            return latest_version[1]
        else:
            print(f"No valid versions found for model {model_name} in S3 bucket")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def upload_model_to_repository(cfg_path, local_model_folder, model_name, is_production):
    """
    Upload a model folder to the S3 model repository with datetime-based folder name.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        local_model_folder (str): The local model folder to be uploaded.
        model_name (str): The model unique name.
        is_production (bool): If True, upload to the production directory; otherwise, upload to the staging directory.

    Returns:
        str: The name of the folder in S3 where the files were uploaded.
    
    Example:
        load_dotenv(resolve_path('.env/.env.development'))
        aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
        upload_model_to_repository(aws_config_path, resolve_path('models/production_model/'), 'tf_trimodel', is_production = False)
    """
    # Determine folder type based on is_production
    folder_type = 'production' if is_production else 'staging'
    
    # Generate the current datetime string
    current_datetime = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    
    # Define S3 folder prefix
    s3_folder_prefix = f"model_repository/{folder_type}/{model_name}/{current_datetime}/"

    # Upload the local folder to S3
    upload_folder_to_s3(cfg_path, local_model_folder, s3_folder_prefix)
    
    return s3_folder_prefix