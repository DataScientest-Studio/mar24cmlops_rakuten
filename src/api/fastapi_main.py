from datetime import timedelta
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import duckdb
import os
from api.utils.make_db import download_initial_db
from api.utils.security import create_access_token, verify_password
from dotenv import load_dotenv
import jwt
from api.utils.resolve_path import resolve_path
from api.utils import predict
from api.utils.predict import load_models_from_file, predict_from_list_models
from contextlib import asynccontextmanager

# Initialize global variables

conn = None
mdl_list = []

# Context manager for lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global aws_config_path, duckdb_path, encrypted_file_path, conn, mdl_list

    # Load environment variables from .env file
    env_path = resolve_path('.env/.env.development')
    load_dotenv(env_path)

    # Convert paths to absolute paths
    aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
    duckdb_path = os.path.join(resolve_path(os.environ['DATA_PATH']), os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
    encrypted_file_path = os.path.join(resolve_path(os.environ['AWS_CONFIG_FOLDER']), '.encrypted')

    # Load model list
    mdl_list = load_models_from_file(aws_config_path, resolve_path('models/model_list.txt'))

    # Check if the DuckDB database file exists locally, if not, download it from S3
    if not os.path.isfile(duckdb_path):
        print('No Database Found locally')
        download_initial_db(aws_config_path, duckdb_path)
        print('Database Successfully Downloaded')

    # Load DuckDB connection   
    conn = duckdb.connect(database=duckdb_path, read_only=False)

    yield

    # Clean up resources
    conn.close()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Model for listing
class Listing(BaseModel):
    description: str
    designation: str
    user_prdtypecode: int
    imageid: int
    
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
    cursor = conn.execute(f"SELECT hashed_password FROM dim_user WHERE username = '{username}'")
    result = cursor.fetchone()
    if not result:
        return False
    
    hashed_password = result[0]
    return verify_password(password, hashed_password)

# Define OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get('/')
def get_index():
    return {'data': 'hello world'}

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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=int(os.environ['ACCESS_TOKEN_EXPIRE_MINUTES']))
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
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
        payload = jwt.decode(token, os.environ['JWT_KEY'], algorithms=[os.environ['ALGORITHM']])
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
    cols = ["designation", "description", 
            "user_prdtypecode", "model_prdtypecode", 
            "waiting_datetime","validate_datetime",
            "status","user","imageid"]
    columns_str = ", ".join(cols)
    cursor = conn.execute(f"SELECT {columns_str} FROM fact_listings WHERE listing_id = {listing_id}")
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    response = dict(zip(cols,result))
    
    return response

# Delete listing endpoint
@app.delete("/delete_listing/{listing_id}")
async def delete_listing(listing_id: int, current_user: dict = Depends(get_current_user)):
    """
    Endpoint to delete a listing.

    Args:
        listing_id (int): The ID of the listing to be deleted.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Message indicating success or failure.
    """
    # Logic to check if user has permission to delete listing
    cursor = conn.execute(f"SELECT user FROM fact_listings WHERE listing_id = {listing_id}")
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")
    if result[0] != current_user["username"]:
        raise HTTPException(status_code=403, detail="This user is not the owner of this listing_id")
    # Logic to delete listing
    conn.execute(f"DELETE FROM fact_listings WHERE listing_id = {listing_id}")
    return {"message": "Listing deleted successfully"}


@app.post("/listing_submit")
async def listing_submit(
    description: str = Form(...),
    designation: str = Form(...),
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Build the SQL query for insertion
    sql_insert = f"""
        INSERT INTO fact_listings (listing_id, description, designation, user, productid, imageid)
        SELECT 
            IFNULL(MAX(listing_id), 0) + 1 AS new_listing_id,
            '{description}' AS description,
            '{designation}' AS designation,
            '{current_user["username"]}' AS username,
            IFNULL(MAX(productid), 0) + 1 AS new_product_id,
            IFNULL(MAX(imageid), 0) + 1 AS new_image_id
        FROM fact_listings;
        SELECT MAX(listing_id), MAX(productid), MAX(imageid) FROM fact_listings;
    """

    # Execute the insertion query
    result = conn.execute(sql_insert).fetchall()[0]

    new_listing_id = result[0]
    new_productid = result[1]
    new_imageid = result[2]
    
    # Save the uploaded image
    image_path = resolve_path(f"data/images/submitted_images/image_{new_imageid}_product_{new_productid}.jpg")
    with open(image_path, "wb") as image_file:
        image_file.write(image.file.read())

    # Predict using the loaded models
    pred = predict_from_list_models(mdl_list, designation, image_path)

    # Construct and return the response
    response = {
        "message": f"Listing added successfully",
        "listing_id": new_listing_id,
        "product_id": new_productid,
        "image_id": new_imageid,
        "prediction": pred
    }
    return response

@app.get('/predict_from_listing')
async def predict_from_listing(listing_id: str):
    
    pred = predict_from_list_models(mdl_list,'Zazie dans le métro est un livre intéressant de Raymond Queneau', resolve_path('data/zazie.jpg'))
    
    return pred

# @app.get('/predict_listing')
# async def predict_from_listing(listing_id: str):
    
#     # Here we can ask a prediction from our models for an already existing listing
    
#     # Si on soumet une image et texte, alors on predit tranquillement
    
#     return predict(model_list, designation, resolve_path(image_path))

# @app.get('/listing_submit')
# async def predict_from_submission(designation: str, image_path: str):
    
#     # Si on propose une annonce, on va chercher l'annonce et on recupere l'image
#     # puis on prédit
    
#     # Si on soumet une image et texte, alors on predit tranquillement
#     return predict(model_list, designation, resolve_path(image_path))

# @app.get('/listing_validate')
# async def predict_from_submission(designation: str, image_path: str):
    
#     # Si on propose une annonce, on va chercher l'annonce et on recupere l'image
#     # puis on prédit
    
#     # Si on soumet une image et texte, alors on predit tranquillement
#     return predict(model_list, designation, resolve_path(image_path))