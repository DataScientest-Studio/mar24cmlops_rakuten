import duckdb
import pandas as pd
from api.utils.s3_utils import create_s3_conn_from_creds, download_from_s3, upload_to_s3
from datetime import datetime
import numpy as np
import os
from passlib.hash import bcrypt
from api.utils.resolve_path import resolve_path


def process_listing(listing_csv_path, prdtypecode_csv_path):
    """
    Process the listing CSV file.

    Args:
    - listing_csv_path (str): Path to the listing CSV file.
    - prdtypecode_csv_path (str): Path to the Y_train CSV file containing prdtypecodes
        from the user
    Returns:
    - listing_df (pd.DataFrame): Processed DataFrame containing listings data.
    """
    listing_df = pd.read_csv(listing_csv_path, index_col=0)
    prdtypecode_df = pd.read_csv(prdtypecode_csv_path, index_col=0)

    listing_df["listing_id"] = listing_df.index
    listing_df = listing_df.join(prdtypecode_df, how="left")

    listing_df["waiting_datetime"] = datetime.now()
    listing_df["validate_datetime"] = datetime.now()
    listing_df["status"] = "validate"
    listing_df["user"] = "init_user"
    listing_df = listing_df.rename(columns={"prdtypecode": "user_prdtypecode"})
    listing_df["model_prdtypecode"] = np.nan

    return listing_df


def init_user_table():
    """
    Initialize the user table with default data including hashed passwords.

    Returns:
    - user_df (pd.DataFrame): DataFrame containing user data.
    """
    user_data = {
        "username": ["jc", "fred", "wilfried", "init_user", "test_user", "test_admin"],
        "first_name": [
            "jc",
            "fred",
            "wilfried",
            "init_user",
            "test_user",
            "test_admin",
        ],
        "access_rights": [
            "administrator",
            "administrator",
            "administrator",
            "user",
            "user",
            "administrator",
        ],
        "hashed_password": [],
    }

    # Hasher les mots de passe
    passwords = ["jc", "fred", "wilfried", "init_user", "test_user", "test_admin"]
    hashed_passwords = [bcrypt.hash(password) for password in passwords]

    user_data["hashed_password"] = hashed_passwords

    user_df = pd.DataFrame(user_data)
    return user_df


def create_table_from_pd_into_duckdb(duckdb_connection, pd_df, table_name):
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
    download_from_s3(s3_conn, "db/rakuten_init.duckdb", local_path)


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
    filename = f"rakuten_{current_datetime}.duckdb"

    # Upload the initialization database file with the datetime-based filename
    upload_to_s3(s3_conn, local_path, f"db/{filename}")


def init_db(duckdb_path, is_test=False):
    """
    Initialize the DuckDB database and save it in duckdb_path (str).
    If is_test argument = True, then will only create fact_listings but with the first 100 listings only
    Returns:
        None
    """
    cfg_path = resolve_path(os.environ["AWS_CONFIG_PATH"])
    data_path = resolve_path(os.environ["DATA_PATH"])
    init_listing_csv_path = os.path.join(data_path, "X_train.csv")
    init_prdtypecode_csv_path = os.path.join(data_path, "Y_train.csv")
    init_dim_prdtypecode_csv_path = os.path.join(data_path, "dim_prdtypecode.csv")

    # Download initial data from S3
    s3_conn = create_s3_conn_from_creds(cfg_path)
    download_from_s3(s3_conn, "X_train.csv", init_listing_csv_path)
    download_from_s3(s3_conn, "Y_train.csv", init_prdtypecode_csv_path)
    download_from_s3(s3_conn, "dim_prdtypecode.csv", init_dim_prdtypecode_csv_path)

    # Process listing data and user data
    listings_df = process_listing(init_listing_csv_path, init_prdtypecode_csv_path)
    user_df = init_user_table()
    dim_prdtypecode = pd.read_csv(init_dim_prdtypecode_csv_path)

    if is_test:
        listings_df = listings_df.head(100)

    # Connect to DuckDB
    duckdb_conn = duckdb.connect(database=duckdb_path, read_only=False)

    # Create and populate tables
    create_table_from_pd_into_duckdb(duckdb_conn, listings_df, "fact_listings")
    create_table_from_pd_into_duckdb(duckdb_conn, user_df, "dim_user")
    create_table_from_pd_into_duckdb(duckdb_conn, dim_prdtypecode, "dim_prdtypecode")

    model_prdtypecode_to_varchar_sql = "ALTER TABLE fact_listings ALTER COLUMN model_prdtypecode SET DATA TYPE VARCHAR;"
    duckdb_conn.execute(model_prdtypecode_to_varchar_sql)
