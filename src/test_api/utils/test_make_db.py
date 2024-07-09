import os
import pytest
import pandas as pd
import duckdb
from passlib.hash import bcrypt
from dotenv import load_dotenv
from api.utils.resolve_path import resolve_path
from api.utils.s3_utils import create_s3_conn_from_creds, download_from_s3
from api.utils.make_db import (
    process_listing,
    init_user_table,
    create_table_from_pd_into_duckdb,
    download_initial_db,
    upload_db,
    init_db,
)


@pytest.fixture(scope="module")
def setup():
    """
    Fixture to setup AWS configuration for tests and create test files.

    Yields:
        dict: A dictionary containing paths and configurations for AWS testing.
    """
    # Setup
    local_test_folder = resolve_path("data/tmp")  # Local folder to upload in tests

    # Load environment variables from .env file
    env_path = resolve_path(".env/.env.development")
    load_dotenv(env_path)

    # Convert paths to absolute paths
    aws_config_path = resolve_path(os.environ["AWS_CONFIG_PATH"])

    # Create test files
    os.makedirs(local_test_folder, exist_ok=True)

    # Create temporary DuckDB database file path
    duckdb_path = os.path.join(local_test_folder, "test_db.duckdb")

    # Create S3 connection
    s3_conn = create_s3_conn_from_creds(aws_config_path)

    # Download test CSV files from S3
    x_extract_path = os.path.join(local_test_folder, "x_extract.csv")
    y_extract_path = os.path.join(local_test_folder, "y_extract.csv")
    download_from_s3(s3_conn, "test/x_extract.csv", x_extract_path)
    download_from_s3(s3_conn, "test/y_extract.csv", y_extract_path)

    yield {
        "cfg_path": resolve_path(os.environ["AWS_CONFIG_PATH"]),
        "data_path": local_test_folder,
        "duckdb_path": duckdb_path,
        "x_extract_path": x_extract_path,
        "y_extract_path": y_extract_path,
    }


@pytest.fixture(scope="module", autouse=True)
def teardown(setup):
    """
    Teardown fixture to clean up resources after tests.

    Args:
        setup (dict): The setup fixture dictionary containing configuration paths and DuckDB paths.
    """
    yield

    # Clean up local files
    for filename in os.listdir(setup["data_path"]):
        file_path = os.path.join(setup["data_path"], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    # Teardown S3
    s3_conn = create_s3_conn_from_creds(setup["cfg_path"])
    bucket_name = "rakutenprojectbucket"
    test_folder_prefix = "test/"

    # List objects in the 'test' folder
    objects = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=test_folder_prefix)
    to_delete = []

    # Files to keep on S3 Test Bucket
    keys_to_ignore = ["test/zazie.jpg", "test/x_extract.csv", "test/y_extract.csv"]

    if "Contents" in objects and objects["Contents"]:
        for obj in objects["Contents"]:
            key = obj["Key"]
            if key not in keys_to_ignore:
                to_delete.append({"Key": key})

        # Delete objects in the 'test' folder
        if to_delete:
            delete_objects_response = s3_conn.delete_objects(
                Bucket=bucket_name, Delete={"Objects": to_delete}
            )

            if "Errors" in delete_objects_response:
                for error in delete_objects_response["Errors"]:
                    print(
                        f"Failed to delete object {error['Key']} from S3: {error['Code']} - {error['Message']}"
                    )
            else:
                print(
                    f"Deleted {len(to_delete)} objects from S3 bucket '{bucket_name}'"
                )
        else:
            print(
                f"No objects to delete in '{test_folder_prefix}' folder in S3 bucket '{bucket_name}'"
            )
    else:
        print(
            f"No objects found in '{test_folder_prefix}' folder in S3 bucket '{bucket_name}'. No objects to delete."
        )


# Test process_listing function
def test_process_listing(setup):
    """
    Test process_listing function to ensure it processes listings correctly.

    This test verifies that the output DataFrame contains necessary columns after processing.
    """
    x_extract_path = setup["x_extract_path"]
    y_extract_path = setup["y_extract_path"]

    df = process_listing(x_extract_path, y_extract_path)
    assert "listing_id" in df.columns
    assert "user_prdtypecode" in df.columns
    assert "validate_datetime" in df.columns
    assert "waiting_datetime" in df.columns
    assert "validate_datetime" in df.columns
    assert "status" in df.columns
    assert "user" in df.columns
    assert "user_prdtypecode" in df.columns
    assert "model_prdtypecode" in df.columns


# Test init_user_table function
def test_init_user_table():
    """
    Test init_user_table function to ensure it initializes user data correctly.

    This test verifies that the user DataFrame contains necessary columns and hashed passwords.
    """
    df = init_user_table()
    assert "username" in df.columns
    assert "hashed_password" in df.columns
    assert bcrypt.verify(
        "jc", df.loc[df["username"] == "jc", "hashed_password"].values[0]
    )


# Test create_table_from_pd_into_duckdb function
def test_create_table_from_pd_into_duckdb(setup):
    """
    Test create_table_from_pd_into_duckdb function to ensure it creates a table in DuckDB from a DataFrame.

    Args:
        setup (dict): The setup fixture dictionary containing configuration paths and DuckDB paths.

    This test verifies that the data is correctly inserted into the DuckDB table.
    """
    conn = duckdb.connect(database=setup["duckdb_path"], read_only=False)
    df = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    create_table_from_pd_into_duckdb(conn, df, "test_table")
    result = conn.execute("SELECT * FROM test_table").df()
    assert len(result) == 2


# Test download_initial_db function
def test_download_initial_db(setup):
    """
    Test download_initial_db function to ensure it downloads the initial database from S3.

    Args:
        setup (dict): The setup fixture dictionary containing configuration paths and DuckDB paths.

    This test verifies that the initial database file is downloaded from S3.
    """
    local_path = os.path.join(setup["data_path"], "rakuten_init.duckdb")
    download_initial_db(setup["cfg_path"], local_path)
    assert os.path.exists(local_path)


# Test upload_db function
def test_upload_db(setup):
    """
    Test upload_db function to ensure it uploads the database to S3 with datetime-based archiving.

    Args:
        setup (dict): The setup fixture dictionary containing configuration paths and DuckDB paths.

    This test verifies that the database file is correctly uploaded to S3.
    """
    local_path = os.path.join(setup["data_path"], "rakuten_test.duckdb")
    open(local_path, "a").close()  # Create an empty test file

    filename = upload_db(setup["cfg_path"], local_path)

    s3_conn = create_s3_conn_from_creds(setup["cfg_path"])
    bucket_name = "rakutenprojectbucket"
    s3_key = f"db/{filename}"
    response = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=s3_key)
    assert "Contents" in response

    s3_conn.delete_objects(
        Bucket=bucket_name, Delete={"Objects": [{"Key": f"db/{filename}"}]}
    )


# Test init_db function
def test_init_db(setup):
    """
    Test init_db function to ensure it initializes the DuckDB database.

    Args:
        setup (dict): The setup fixture dictionary containing configuration paths and DuckDB paths.

    This test verifies that the database is initialized with the correct number of records.
    """
    init_db(setup["duckdb_path"], is_test=True)
    conn = duckdb.connect(database=setup["duckdb_path"], read_only=True)
    result = conn.execute("SELECT COUNT(*) FROM fact_listings").fetchone()[0]
    assert result == 100  # Assuming the initial test DB contains 100 listings
