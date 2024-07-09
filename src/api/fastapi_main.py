from datetime import timedelta, datetime
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import duckdb
import os
from api.utils.make_db import download_initial_db
from api.utils.security import create_access_token, verify_password
from dotenv import load_dotenv
import jwt
from api.utils.resolve_path import resolve_path
from api.utils.get_models import verify_and_download_models
from api.utils.predict import load_models_from_file, predict_from_list_models
from api.utils.s3_utils import download_from_s3, create_s3_conn_from_creds
from contextlib import asynccontextmanager
import json
from api.utils.write_logs import log_user_action, log_product_action

# Context manager for lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.

    This function initializes the necessary environment variables, sets up the database connection,
    and loads the machine learning models required by the application. It ensures that resources
    are properly cleaned up when the application is shutting down.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None

    Note:
        The following global variables are defined and used:
            - aws_config_path: Path to AWS configuration.
            - duckdb_path: Path to the DuckDB database file.
            - encrypted_file_path: Path to the encrypted file.
            - conn: DuckDB connection object.
            - mdl_list: List of loaded machine learning models.
    """
    global aws_config_path, duckdb_path, encrypted_file_path, conn, mdl_list, s3_conn

    # Load environment variables from .env file
    env_path = resolve_path(".env/.env.development")
    load_dotenv(env_path)

    # Convert paths to absolute paths
    aws_config_path = resolve_path(os.environ["AWS_CONFIG_PATH"])
    duckdb_path = os.path.join(
        resolve_path(os.environ["DATA_PATH"]), os.environ["RAKUTEN_DB_NAME"].lstrip("/")
    )
    encrypted_file_path = os.path.join(
        resolve_path(os.environ["AWS_CONFIG_FOLDER"]), ".encrypted"
    )

    # Verify and download models if necessary
    verify_and_download_models(aws_config_path, resolve_path("models/model_list.txt"))

    # Load model list
    mdl_list = load_models_from_file(
        aws_config_path, resolve_path("models/model_list.txt")
    )

    # Check if the DuckDB database file exists locally, if not, download it from S3
    if not os.path.isfile(duckdb_path):
        print("No Database Found locally")
        download_initial_db(aws_config_path, duckdb_path)
        print("Database Successfully Downloaded")

    # Load DuckDB connection
    conn = duckdb.connect(database=duckdb_path, read_only=False)

    # Créer une connexion S3
    s3_conn = create_s3_conn_from_creds(aws_config_path)

    yield

    # Clean up resources
    conn.close()
    s3_conn.close()

    # upload_db
    # upload_db(aws_config_path, duckdb_path)


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


# Model for listing
class Listing(BaseModel):
    description: str
    designation: str


class ListingWithImage(Listing):
    image: UploadFile = File(...)


class ValidateListing(BaseModel):
    listing_id: int
    user_prdtypecode: int


# Auth User function
def authenticate_user(username: str, password: str):
    """
    Authenticate a user with username and password.

    Args:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        bool: True if the user is authenticated, False otherwise.
    """
    cursor = conn.execute(
        f"SELECT hashed_password FROM dim_user WHERE username = '{username}'"
    )
    result = cursor.fetchone()
    if not result:
        return False

    hashed_password = result[0]
    return verify_password(password, hashed_password)

# Get User Rights function
def get_user_access_rights(username: str):
    """
    Get user access rights from a username.

    Args:
        username (str): The username of the user.

    Returns:
        str: Access Rights from the username
    """
    cursor = conn.execute(
        f"SELECT access_rights FROM dim_user WHERE username = '{username}'"
    )
    access_right = cursor.fetchone()[0]
    return access_right


# Define OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/")
def get_index():
    return {"data": "hello world"}


# Endpoint to authenticate and get token
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint to authenticate user and provide access token.

    Args:
        form_data (OAuth2PasswordRequestForm): The form data containing username and password.

    Returns:
        dict: Dictionary containing the access token and token type.
    """
    username = form_data.username
    password = form_data.password
    if not authenticate_user(username, password):
        
        log_user_action('/token', username, 401, None)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        
    access_rights = get_user_access_rights(username)
    access_token_expires = timedelta(
        minutes=int(os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"])
    )
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    
    log_user_action('/token', username, 200, access_rights)
    
    return {"access_token": access_token, "token_type": "bearer"}


# Dependency function to get current user from token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get the current user from the token.

    Args:
        token (str): The JWT token.

    Returns:
        dict: Dictionary containing the username.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, os.environ["JWT_KEY"], algorithms=[os.environ["ALGORITHM"]]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = {"username": username}
    except jwt.JWTError:
        raise credentials_exception
    return token_data


# Read listing endpoint
@app.get("/read_listing/{listing_id}")
async def read_listing(listing_id: int, current_user: dict = Depends(get_current_user)):
    """
    Endpoint to read listing description.

    Args:
        listing_id (int): The ID of the listing.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Dictionary containing listing description.
    """
    # Dummy logic to retrieve listing description based on listing_id
    cols = [
        "designation",
        "description",
        "productid",
        "imageid",
        "user_prdtypecode",
        "model_prdtypecode",
        "waiting_datetime",
        "validate_datetime",
        "status",
        "user",
        "imageid",
    ]
    columns_str = ", ".join(cols)
    cursor = conn.execute(
        f"SELECT {columns_str} FROM fact_listings WHERE listing_id = {listing_id}"
    )
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")

    response = dict(zip(cols, result))

    return response


# Delete listing endpoint
@app.delete("/delete_listing/{listing_id}")
async def delete_listing(
    listing_id: int, current_user: dict = Depends(get_current_user)
):
    """
    Endpoint to delete a listing.

    Args:
        listing_id (int): The ID of the listing to be deleted.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Message indicating success or failure.
    """
    # Logic to check if user has permission to delete listing
    cursor = conn.execute(
        f"SELECT user FROM fact_listings WHERE listing_id = {listing_id}"
    )
    result = cursor.fetchone()
    if not result:
        log_product_action("delete_listing", 404, current_user["username"], listing_id)
        raise HTTPException(status_code=404, detail="Listing not found")
    if result[0] != current_user["username"]:
        log_product_action("delete_listing", 403, current_user["username"], listing_id)
        raise HTTPException(
            status_code=403, detail="This user is not the owner of this listing_id"
        )
    # Logic to delete listing
    conn.execute(f"DELETE FROM fact_listings WHERE listing_id = {listing_id}")
    log_product_action("delete_listing", 200, current_user["username"], listing_id)
    return {"message": "Listing deleted successfully"}


@app.post("/listing_submit")
async def listing_submit(
    listing: ListingWithImage = Depends(),
    current_user: dict = Depends(get_current_user),
):
    """
    Endpoint to submit a new listing along with an image.

    This function handles the insertion of a new listing into the database,
    saves the uploaded image, and makes a prediction using the loaded models.

    Args:
        listing (ListingWithImage): The listing information along with the uploaded image.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Response containing the listing ID, product ID, image ID, and prediction result.
    """
    # Build the SQL query for insertion

    waiting_datetime = datetime.now()

    sql_insert = f"""
        INSERT INTO fact_listings (listing_id, description, designation, user, productid, imageid, status, waiting_datetime)
        SELECT 
            IFNULL(MAX(listing_id), 0) + 1 AS new_listing_id,
            '{listing.description}' AS description,
            '{listing.designation}' AS designation,
            '{current_user["username"]}' AS username,
            IFNULL(MAX(productid), 0) + 1 AS new_product_id,
            IFNULL(MAX(imageid), 0) + 1 AS new_image_id,
            'waiting' AS status,
            '{waiting_datetime}' AS waiting_datetime
        FROM fact_listings;
        SELECT MAX(listing_id), MAX(productid), MAX(imageid) FROM fact_listings;
    """

    # Execute the insertion query
    result = conn.execute(sql_insert).fetchall()[0]

    new_listing_id = result[0]
    new_productid = result[1]
    new_imageid = result[2]

    # Save the uploaded image
    image_path = resolve_path(
        f"data/images/submitted_images/image_{new_imageid}_product_{new_productid}.jpg"
    )
    with open(image_path, "wb") as image_file:
        image_file.write(listing.image.file.read())

    # Predict using the loaded models
    pred = predict_from_list_models(mdl_list, listing.designation, image_path)
    # Convert prediction dictionary to a JSON string
    pred_json = json.dumps(pred)

    # Inserting modele pred into table
    sql_prediction_insert = f"UPDATE fact_listings SET model_prdtypecode = {pred_json} WHERE listing_id = {new_listing_id}"
    conn.execute(sql_prediction_insert)
    
    # Log
    log_product_action("listing_submit", 200, current_user["username"], new_listing_id, pred_json)
    
    # Construct and return the response
    response = {
        "message": "Listing added successfully",
        "listing_id": new_listing_id,
        "product_id": new_productid,
        "image_id": new_imageid,
        "prediction": pred,
    }
    return response


@app.post("/listing_validate")
async def listing_validate(
    validation: ValidateListing = Depends(),
    current_user: dict = Depends(get_current_user),
):
    """
    Endpoint to validate a listing by inserting user_prdtypecode.

    Args:
        validation (ValidateListing): The listing ID and user_prdtypecode to be inserted.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Message indicating success or failure.
    """
    # Check if user has permission to validate listing
    cursor = conn.execute(
        f"SELECT user FROM fact_listings WHERE listing_id = {validation.listing_id}"
    )
    result = cursor.fetchone()
    if not result:
        log_product_action("listing_validate", 404, current_user["username"], validation.listing_id)
        raise HTTPException(status_code=404, detail="Listing not found")
    if result[0] != current_user["username"]:
        log_product_action("listing_validate", 403, current_user["username"], validation.listing_id)
        raise HTTPException(
            status_code=403, detail="This user is not the owner of this listing_id"
        )

    # Insert into table the user_prdtypecode
    try:
        validate_datetime = datetime.now()

        conn.execute(
            f"UPDATE fact_listings SET user_prdtypecode = {validation.user_prdtypecode}, status = 'validate', validate_datetime = '{validate_datetime}' WHERE listing_id = {validation.listing_id}"
        )
        log_product_action("listing_validate", 200, current_user["username"], validation.listing_id, None, validation.user_prdtypecode)
    except Exception as e:
        log_product_action("listing_validate", 500, current_user["username"], validation.listing_id)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    return {
        "message": f"Listing {validation.listing_id} validated successfully with user_prdtypecode {validation.user_prdtypecode}"
    }


@app.post("/predict_listing")
async def predict_listing(
    listing_id: int, current_user: dict = Depends(get_current_user)
):
    """
    Endpoint to predict and upload image to S3 for a given listing.

    This function retrieves the productid and imageid based on listing_id,
    downloads the corresponding image locally, and uploads it to S3.

    Args:
        listing_id (int): The ID of the listing to predict and upload.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Response containing the message indicating success or failure.
    """
    cursor = conn.execute(
        f"SELECT designation, productid, imageid FROM fact_listings WHERE listing_id = {listing_id}"
    )
    result = cursor.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")

    designation, productid, imageid = result[0], result[1], result[2]
    image_path = resolve_path(
        f"data/images/submitted_images/image_{imageid}_product_{productid}.jpg"
    )

    if not os.path.isfile(image_path):
        try:
            # Download from S3
            download_from_s3(
                s3_conn,
                f"image_train/image_{imageid}_product_{productid}.jpg",
                image_path,
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while downloading from S3: {e}",
            )

    pred = predict_from_list_models(mdl_list, designation, image_path)

    # Construct and return the response
    response = {
        "message": "Listing predicted successfully",
        "listing_id": listing_id,
        "product_id": productid,
        "image_id": imageid,
        "prediction": pred,
    }
    return response
