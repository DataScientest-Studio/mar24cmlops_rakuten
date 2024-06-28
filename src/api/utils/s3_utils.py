import boto3
import configparser
import os
from src.api.utils.security import decrypt_file
from concurrent.futures import ThreadPoolExecutor

def load_aws_cfg(file_path):
    """
    Load AWS credentials and role parameters from a configuration file.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing 'aws_access_key_id', 'aws_secret_access_key', 'role_arn', and 'role_session_name'.
    """
    decr_file = os.path.join(os.environ['AWS_CONFIG_FOLDER'],".aws_config_decr.ini")
    decrypt_file(os.environ['KEY'], file_path, decr_file)
    
    config = configparser.ConfigParser()
    config.read(decr_file)
    aws_cfg = {
        'aws_access_key_id': config.get('default', 'aws_access_key_id'),
        'aws_secret_access_key': config.get('default', 'aws_secret_access_key'),
        'role_arn': config.get('default', 'role_arn'),
        'role_session_name': config.get('default', 'role_session_name')
    }
    return aws_cfg

def assume_role(aws_access_key_id, aws_secret_access_key, role_arn, role_session_name):
    """
    Assume an IAM role and obtain temporary credentials.

    Args:
        aws_access_key_id (str): The AWS access key ID.
        aws_secret_access_key (str): The AWS secret access key.
        role_arn (str): The ARN of the role to assume.
        role_session_name (str): The name of the role session.

    Returns:
        dict: A dictionary containing temporary credentials.
    """
    client = boto3.client(
        'sts',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    response = client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=role_session_name
    )
    return response['Credentials']

def create_sts_session(credentials):
    """
    Create a Boto3 session with temporary credentials.

    Args:
        credentials (dict): A dictionary containing 'AccessKeyId', 'SecretAccessKey', and 'SessionToken'.

    Returns:
        boto3.Session: A Boto3 session initialized with the temporary credentials.
    """
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )

def create_s3_conn_from_creds(cfg):
    
    aws_cfg = load_aws_cfg(cfg)
    
    credentials = assume_role(
        aws_cfg['aws_access_key_id'], 
        aws_cfg['aws_secret_access_key'], 
        aws_cfg['role_arn'], 
        aws_cfg['role_session_name']
        )
    
    sts_session = create_sts_session(credentials)
    
    return sts_session.client('s3')
    
def download_from_s3(s3_conn ,bucket_path, local_path):
    """
    Download a file from an S3 bucket to a local path.

    Args:
        bucket_path (str): The path to the file in the S3 bucket, relative to the bucket root.
        local_path (str): The local path where the file should be downloaded.

    """
    # Download the file from S3
    s3_conn.download_file("rakutenprojectbucket", bucket_path, local_path)
    
def upload_to_s3(s3_conn, local_path, bucket_path):
    """
    Upload a file from a local path to an S3 bucket.

    Args:
        local_path (str): The local path of the file to be uploaded.
        bucket_path (str): The path in the S3 bucket where the file should be uploaded, relative to the bucket root.
    """
    # Upload the file to S3
    s3_conn.upload_file(local_path, "rakutenprojectbucket", bucket_path)

def upload_file_list_to_s3(s3_conn, local_path_x_bucket_path_tuple):
    """
    Upload a list of files to an S3 bucket in parallel.

    Args:
        s3_conn (boto3.Client): The S3 client.
        local_path_x_bucket_path_tuple (list of tuple): A list of tuples where each tuple contains the local path and the bucket path.
    """
    with ThreadPoolExecutor() as executor:
        executor.map(lambda local_x_bucket: upload_to_s3(s3_conn, local_x_bucket[0], local_x_bucket[1]), local_path_x_bucket_path_tuple)

# if __name__ == "__main__":
    
#     cfg_path = '/mnt/c/Users/cjean/Documents/workspace/mar24cmlops_rakuten/.aws/.aws_config.ini'
#     s3_conn = create_s3_conn_from_creds(cfg_path)
    
#     download_from_s3(
#         s3_conn = s3_conn,
#         bucket_path = 'imgtest.jpg',
#         local_path = '/mnt/c/Users/cjean/Documents/workspace/mar24cmlops_rakuten/data/imgtest.jpg'
#     )

#     upload_to_s3(
#         s3_conn = s3_conn, 
#         local_path = '/mnt/c/Users/cjean/Documents/workspace/mar24cmlops_rakuten/data/imgtest.jpg',
#         bucket_path = 'imgtest.jpg'
#     )

# from src.api.utils.s3_utils import upload_file_list_to_s3, create_s3_conn_from_creds
# import os
# from dotenv import load_dotenv

# images = os.listdir('data/images/image_test')
# to_upload_images = images

# images_path = [os.path.join('data/images/image_test',image) for image in to_upload_images]
# bucket_path = [os.path.join('image_test_2/',image) for image in to_upload_images]

# local_x_bucket = list(zip(images_path,bucket_path))

# # Load environment variables from .env file
# load_dotenv('.env/.env.development')
# aws_config_path = os.environ['AWS_CONFIG_PATH']

# s3_conn = create_s3_conn_from_creds(aws_config_path)
# upload_file_list_to_s3(s3_conn, local_x_bucket)