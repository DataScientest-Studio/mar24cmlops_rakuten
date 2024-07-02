import pytest
import os
import boto3
from api.utils.security import decrypt_file
from api.utils.resolve_path import resolve_path
from dotenv import load_dotenv
from api.utils.s3_utils import load_aws_cfg, create_s3_conn_from_creds, download_from_s3, upload_to_s3, upload_file_list_to_s3, upload_folder_to_s3

@pytest.fixture(scope='module')
def setup():
    """
    Fixture to setup AWS configuration for tests.

    Yields:
        dict: A dictionary containing paths and configurations for AWS testing.
    """
    # Setup

    local_test_folder = 'data/tmp'  # Local folder to upload in tests
    s3_test_folder = 'test'  # S3 folder name for testing
    
    # Load environment variables from .env file
    env_path = resolve_path('.env/.env.development')
    load_dotenv(env_path)

    # Convert paths to absolute paths
    aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
    duckdb_path = os.path.join(resolve_path(os.environ['DATA_PATH']), os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
    encrypted_file_path = os.path.join(resolve_path(os.environ['AWS_CONFIG_FOLDER']), '.encrypted')

    yield {
        'cfg_path': aws_config_path,
        'local_test_folder': local_test_folder,
        's3_test_folder': s3_test_folder
    }

@pytest.fixture(scope='module', autouse=True)
def teardown():
    """
    Fixture to teardown AWS resources after tests.
    This fixture deletes uploaded test files and folders from the 'test' folder in S3 bucket 'rakutenprojectbucket'.
    """
    to_delete = []

    yield

    # Teardown
    s3_conn = boto3.client('s3')
    bucket_name = 'rakutenprojectbucket'
    test_folder_prefix = 'test/'

    # List objects in the 'test' folder
    objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=test_folder_prefix)

    if 'Contents' in objects and objects['Contents']:
        for obj in objects['Contents']:
            key = obj['Key']
            to_delete.append({'Bucket': bucket_name, 'Key': key})
        
        # Delete objects in the 'test' folder
        if to_delete:
            delete_objects_response = s3_conn.delete_objects(Bucket=bucket_name, Delete={'Objects': to_delete})

            if 'Errors' in delete_objects_response:
                for error in delete_objects_response['Errors']:
                    print(f"Failed to delete object {error['Key']} from S3: {error['Code']} - {error['Message']}")
    else:
        print(f"No objects found in '{test_folder_prefix}' folder in S3 bucket '{bucket_name}'. No objects to delete.")

def test_load_aws_cfg(setup):
    """
    Test function for loading AWS configuration from a config file.
    """
    cfg = load_aws_cfg(setup['cfg_path'])
    assert isinstance(cfg, dict)
    assert 'aws_access_key_id' in cfg
    assert 'aws_secret_access_key' in cfg
    assert 'role_arn' in cfg
    assert 'role_session_name' in cfg

def test_create_s3_conn_from_creds(setup):
    """
    Test function for creating an S3 connection from AWS credentials.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    assert s3_conn is not None

def test_download_from_s3(setup):
    """
    Test function for downloading a file from S3.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    local_path = '/tmp/downloaded_file.jpg'  # Temporary local path for download
    bucket_path = 'test/downloaded_file.jpg'  # Path to file in S3 bucket
    download_from_s3(s3_conn, bucket_path, local_path)
    assert os.path.exists(local_path)

def test_upload_to_s3(setup):
    """
    Test function for uploading a file to S3.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    local_path = '/path/to/local/file.txt'  # Path to a local file to upload
    bucket_path = 'test/uploaded_file.txt'  # Path in S3 bucket for upload
    upload_to_s3(s3_conn, local_path, bucket_path)

def test_upload_file_list_to_s3(setup):
    """
    Test function for uploading a list of files to S3.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    local_path_x_bucket_path_tuple = [('/path/to/local/file1.txt', 'test/file1.txt'),
                                      ('/path/to/local/file2.txt', 'test/file2.txt')]
    upload_file_list_to_s3(s3_conn, local_path_x_bucket_path_tuple)

def test_upload_folder_to_s3(setup):
    """
    Test function for uploading a folder to S3.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    local_folder = setup['local_test_folder']
    s3_folder = setup['s3_test_folder']
    uploaded_folder_name = upload_folder_to_s3(setup['cfg_path'], local_folder, s3_folder)
    # Check if folder is uploaded and other assertions
    assert uploaded_folder_name == s3_folder

def test_check_db_folder_exists_and_download(setup):
    """
    Test function to check if 'db/' folder exists on S3 and download 'rakuten_init.duckdb' and 'rakuten_test.duckdb' files.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    bucket_name = 'rakutenprojectbucket'
    db_folder_path = 'db/'

    # Check if db/ folder exists on S3
    objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=db_folder_path)
    assert 'Contents' in objects

    # Check if 'rakuten_init.duckdb' file exists in db/ folder
    init_file_path = db_folder_path + 'rakuten_init.duckdb'
    try:
        s3_conn.head_object(Bucket=bucket_name, Key=init_file_path)
    except s3_conn.exceptions.ClientError as e:
        assert False, f"'rakuten_init.duckdb' not found in S3 bucket: {e}"

    # Check if 'rakuten_test.duckdb' file exists in db/ folder
    test_file_path = db_folder_path + 'rakuten_test.duckdb'
    try:
        s3_conn.head_object(Bucket=bucket_name, Key=test_file_path)
    except s3_conn.exceptions.ClientError as e:
        assert False, f"'rakuten_test.duckdb' not found in S3 bucket: {e}"

    # Download 'rakuten_init.duckdb' file
    local_init_path = '/tmp/rakuten_init.duckdb'
    download_from_s3(s3_conn, init_file_path, local_init_path)
    assert os.path.exists(local_init_path)

    # Download 'rakuten_test.duckdb' file
    local_test_path = '/tmp/rakuten_test.duckdb'
    download_from_s3(s3_conn, test_file_path, local_test_path)
    assert os.path.exists(local_test_path)

def test_check_model_repository(setup):
    """
    Test function to check if 'model_repository/' folder exists on S3 and contains expected subfolders and files.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    bucket_name = 'rakutenprojectbucket'
    model_repo_path = 'model_repository/'

    # Check if model_repository/ folder exists on S3
    objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=model_repo_path)
    assert 'Contents' in objects

    # Check if production/ subfolder exists
    production_path = model_repo_path + 'production/'
    try:
        s3_conn.head_object(Bucket=bucket_name, Key=production_path)
    except s3_conn.exceptions.ClientError as e:
        assert False, f"'production/' folder not found in S3 bucket: {e}"

    # Check if staging/ subfolder exists
    staging_path = model_repo_path + 'staging/'
    try:
        s3_conn.head_object(Bucket=bucket_name, Key=staging_path)
    except s3_conn.exceptions.ClientError as e:
        assert False, f"'staging/' folder not found in S3 bucket: {e}"

    # Check if production/init_model/ folder exists
    init_model_path = production_path + 'init_model/'
    try:
        s3_conn.head_object(Bucket=bucket_name, Key=init_model_path)
    except s3_conn.exceptions.ClientError as e:
        assert False, f"'init_model/' folder not found in 'production/' folder: {e}"

    # Check if production/init_model/ folder is not empty
    objects_in_init_model = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=init_model_path)
    assert 'Contents' in objects_in_init_model