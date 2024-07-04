import pytest
import os
from api.utils.resolve_path import resolve_path
from dotenv import load_dotenv
from api.utils.s3_utils import load_aws_cfg, create_s3_conn_from_creds, download_from_s3, upload_to_s3, upload_file_list_to_s3, upload_folder_to_s3

@pytest.fixture(scope='module')
def setup():
    """
    Fixture to setup AWS configuration for tests and create test files.

    Yields:
        dict: A dictionary containing paths and configurations for AWS testing.
    """
    # Setup
    local_test_folder = resolve_path('data/tmp')  # Local folder to upload in tests
    s3_test_folder = 'test'  # S3 folder name for testing

    # Load environment variables from .env file
    env_path = resolve_path('.env/.env.development')
    load_dotenv(env_path)

    # Convert paths to absolute paths
    aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])

    # Create test files
    os.makedirs(local_test_folder, exist_ok=True)
    testfile1_path = os.path.join(local_test_folder, 'testfile1.txt')
    testfile2_path = os.path.join(local_test_folder, 'testfile2.txt')

    with open(testfile1_path, 'w') as f:
        f.write('test1')
    
    with open(testfile2_path, 'w') as f:
        f.write('test2')

    yield {
        'cfg_path': aws_config_path,
        'local_test_folder': local_test_folder,
        's3_test_folder': s3_test_folder,
        'testfile1_path': testfile1_path,
        'testfile2_path': testfile2_path
    }

    # Cleanup local test files except README.md
    for filename in os.listdir(local_test_folder):
        file_path = os.path.join(local_test_folder, filename)
        if filename != 'README.md':
            os.remove(file_path)

@pytest.fixture(scope='module', autouse=True)
def teardown(setup):
    """
    Fixture to teardown AWS resources after tests.
    This fixture deletes uploaded test files and folders from the 'test' folder in S3 bucket 'rakutenprojectbucket',
    except for 'test/zazie.jpg'.
    """
    to_delete = []
    
    yield

    # Teardown
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    bucket_name = 'rakutenprojectbucket'
    test_folder_prefix = 'test/'

    # List objects in the 'test' folder
    objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=test_folder_prefix)

    if 'Contents' in objects and objects['Contents']:
        for obj in objects['Contents']:
            key = obj['Key']
            if key != 'test/zazie.jpg':
                to_delete.append({'Key': key})

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
    local_path = os.path.join(setup['local_test_folder'], 'zazie.jpg')  # Temporary local path for download
    bucket_path = 'test/zazie.jpg'  # Path to file in S3 bucket
    download_from_s3(s3_conn, bucket_path, local_path)
    assert os.path.exists(local_path)

def test_upload_to_s3(setup):
    """
    Test function for uploading a file to S3.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    local_path = setup['testfile1_path']  # Path to a local file to upload
    bucket_path = 'test/uploaded_file.txt'  # Path in S3 bucket for upload
    upload_to_s3(s3_conn, local_path, bucket_path)

def test_upload_file_list_to_s3(setup):
    """
    Test function for uploading a list of files to S3.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    local_path_x_bucket_path_tuple = [(setup['testfile1_path'], 'test/file1.txt'),
                                      (setup['testfile2_path'], 'test/file2.txt')]
    upload_file_list_to_s3(s3_conn, local_path_x_bucket_path_tuple)

def test_upload_folder_to_s3(setup): # A verifier pour le return
    """
    Test function for uploading a folder to S3.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    local_folder = setup['local_test_folder']
    s3_folder = setup['s3_test_folder']
    uploaded_folder_name = upload_folder_to_s3(setup['cfg_path'], local_folder, s3_folder)
    # Check if folder is uploaded and other assertions
    assert uploaded_folder_name == s3_folder

def test_check_db_folder_exists(setup):
    """
    Test function to check if 'db/' folder exists on S3 and contains 'rakuten_init.duckdb' and 'rakuten_test.duckdb' files.
    """
    s3_conn = create_s3_conn_from_creds(setup['cfg_path'])
    bucket_name = 'rakutenprojectbucket'
    db_folder_path = 'db/'

    # Check if db/ folder exists on S3
    objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=db_folder_path)
    assert 'Contents' in objects, f"'{db_folder_path}' folder not found in S3 bucket"

    # Check if 'rakuten_init.duckdb' and 'rakuten_test.duckdb' files exist in db/ folder
    expected_files = ['rakuten_init.duckdb', 'rakuten_test.duckdb']
    for file_name in expected_files:
        file_path = db_folder_path + file_name
        file_objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=file_path)
        assert 'Contents' in file_objects and any(obj['Key'] == file_path for obj in file_objects['Contents']), f"'{file_name}' not found in S3 bucket"

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
    production_objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=production_path)
    assert 'Contents' in production_objects, f"'{production_path}' folder not found in S3 bucket"

    # Check if staging/ subfolder exists
    staging_path = model_repo_path + 'staging/'
    staging_objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=staging_path)
    assert 'Contents' in staging_objects, f"'{staging_path}' folder not found in S3 bucket"

    # Check if production/init_model/ folder exists
    init_model_path = production_path + 'init_model/'
    init_model_objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=init_model_path)
    assert 'Contents' in init_model_objects, f"'{init_model_path}' folder not found in S3 bucket"

    # Check if production/init_model/ folder is not empty
    assert len(init_model_objects['Contents']) > 0, f"'{init_model_path}' folder is empty in S3 bucket"