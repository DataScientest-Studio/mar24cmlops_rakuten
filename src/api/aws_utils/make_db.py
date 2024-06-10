import duckdb
import pandas as pd
from aws_utils.s3_utils import create_s3_conn_from_creds, download_from_s3, upload_to_s3
from datetime import datetime
import numpy as np
import os

def process_listing(listing_csv_path):
    """
    Process the listing CSV file.

    Args:
    - listing_csv_path (str): Path to the listing CSV file.

    Returns:
    - listing_df (pd.DataFrame): Processed DataFrame containing listings data.
    """
    listing_df = pd.read_csv(listing_csv_path, index_col= 0)
    listing_df['listing_id'] = listing_df.index
    listing_df = listing_df.rename(columns={'productid': 'user_prdtypecode'})
    listing_df['model_prdtypecode'] = np.nan
    listing_df['waiting_datetime'] = datetime.now()
    listing_df['validate_datetime'] = datetime.now()
    listing_df['status'] = 'validate'
    listing_df['user'] = 'init_user'
    return(listing_df)

def init_user_table():
    """
    Initialize the user table with default data.

    Returns:
    - user_df (pd.DataFrame): DataFrame containing user data.
    """
    user_data = {
    'username': ['jc','fred','wilfried','init_user'],
    'first_name': ['jc','fred','wilfried','init_user'],
    'hashed_password': ['jc','fred','wilfried','init_user'],
    'access_rights': ['administrator','administrator','administrator','user']
    }
    user_df = pd.DataFrame(user_data)
    return(user_df)
    
def create_table_from_pd_into_duckdb(duckdb_connection,pd_df, table_name):
    """
    Loads a CSV file into a DuckDB database.

    Args:
    - duckdb_connection (duckdb): The DuckDB connection.
    - pd_df (pd.DataFrame): The pd.DataFrame to be loaded.
    - table_name (str): The name of the table in DuckDB.

    Returns:
        None
    """
    duckdb_connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM pd_df")

def save_duckdb_to_parquet(duckdb_conn, db_file_path):
    """
    Saves the DuckDB database to a parquetfile.

    Args:
    - duckdb_conn: The DuckDB connection.
    - db_file_path (str): The path where the database will be saved.
    
    Returns:
        None
    """
    duckdb_conn.execute(f"EXPORT DATABASE '{db_file_path}' (FORMAT 'PARQUET')")

def download_initial_db(cfg_path, local_path):
    """
    Download the initialization database file from the S3 bucket.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        local_path (str): The local path where the database file should be downloaded.
    """
    # Create an S3 connection
    s3_conn = create_s3_conn_from_creds(cfg_path)
    
    # Download the initialization database file
    download_from_s3(s3_conn, 'db/rakuten_init.duckdb', local_path)

def upload_db(cfg_path, local_path):
    """
    Upload the database file to the S3 bucket with datetime-based archiving.

    Args:
        cfg_path (str): The path to the AWS configuration file.
        local_path (str): The local path of the database file to be uploaded.
    """
    # Create an S3 connection
    s3_conn = create_s3_conn_from_creds(cfg_path)
    
    # Create a filename with datetime-based archiving
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'rakuten_init_{current_datetime}.duckdb'
    
    # Upload the initialization database file with the datetime-based filename
    upload_to_s3(s3_conn, local_path, f'db/{filename}')

def init_db(duckdb_path):
    
    cfg_path = '/mnt/c/Users/cjean/Documents/workspace/mar24cmlops_rakuten/.aws/.aws_config.ini'
    s3_conn = create_s3_conn_from_creds(cfg_path)
    download_from_s3(s3_conn,'X_train.csv','/mnt/c/Users/cjean/Documents/workspace/mar24cmlops_rakuten/data/X_train.csv')

    listings_df = process_listing('/mnt/c/Users/cjean/Documents/workspace/mar24cmlops_rakuten/data/X_train.csv')
    duckdb_conn = duckdb.connect(database=duckdb_path, read_only=False)
    create_table_from_pd_into_duckdb(duckdb_conn, listings_df, 'fact_listings')

    user_df = init_user_table()
    create_table_from_pd_into_duckdb(duckdb_conn, user_df, 'dim_user')

# init_db('/mnt/c/Users/cjean/Documents/workspace/mar24cmlops_rakuten/data/rakuten_db.duckdb')

# Example Usage 

# (Init of Rakuten DB, saved on S3)

# listing_df = process_listing('X_train.csv')
# user_df = init_user_table()
# db_file_path = '/home/jc/Workspace/mar24cmlops_rakuten/data/rakuten_db.duckdb'
# con = duckdb.connect(database=db_file_path, read_only=False)
# create_table_from_pd_into_duckdb(con, listing_df, 'fact_listings')
# create_table_from_pd_into_duckdb(con, user_df, 'dim_user')
# con.close()

# (Download Rakuten DB from S3)
# download_db_from_s3('/home/jc/Workspace/mar24cmlops_rakuten/.aws/.aws_config', 
#                     'rakuten_db.duckdb',
#                     'rakutenprojectbucket',
#                     '/home/jc/Workspace/mar24cmlops_rakuten/rakuten_db.duckdb')

# (Upload Rakuten DB from S3)
# upload_db_to_s3('/home/jc/Workspace/mar24cmlops_rakuten/.aws/.aws_config', 
#                 '/home/jc/Workspace/mar24cmlops_rakuten/data/rakuten_db.duckdb',
#                 'rakutenprojectbucket')