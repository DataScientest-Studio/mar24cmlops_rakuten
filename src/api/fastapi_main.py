from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import duckdb
import uvicorn
import os
from aws_utils.make_db import download_initial_db
from dotenv import load_dotenv
from passlib.hash import bcrypt
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "7857d2e32966c142dd14307856b540a4c9ee94fc7af45078a82762c74c33c6b5"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Model for listing
class Listing(BaseModel):
    description: str
    designation: str
    user_prdtypecode: int
    imageid: int

# Define OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password verification function
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
    return bcrypt.verify(password, hashed_password)

# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def get_index():
    return {'data': 'hello world'}

# Endpoint to authenticate and get token
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    if not authenticate_user(username, password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Dependency function to get current user from token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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

# Add listing endpoint
@app.post("/add_listing")
async def add_listing(listing: Listing, current_user: dict = Depends(get_current_user)):
    """
    Endpoint to add a new listing.

    Args:
        listing (Listing): Listing information to be added.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Message indicating success or failure.
    """
    sql = f"""
            INSERT INTO fact_listings (listing_id, description, user)
            SELECT IFNULL(MAX(listing_id), 0) + 1, '{listing.description}', '{current_user["username"]}'
            FROM fact_listings;
            SELECT MAX(listing_id) FROM fact_listings;
        """ 
    
    listing_id_added = conn.execute(sql).fetchall()[0][0]

    return {"message": f"Listing {listing_id_added} added successfully"}

if __name__ == "__main__":
    load_dotenv('.env/.env.development')
    
    aws_config_path = os.environ['AWS_CONFIG_PATH']
    duckdb_path = os.path.join(os.environ['DATA_PATH'], os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
    rakuten_db_name = os.environ['RAKUTEN_DB_NAME']
    
    if not os.path.isfile(duckdb_path):
        print('No Database Found locally')
        # Since no database found for the API, download the initial database from S3
        download_initial_db(aws_config_path, duckdb_path)
        print('Database Sucessfully Downloaded')
        
    # Load DuckDB connection   
    conn = duckdb.connect(database=duckdb_path, read_only=False)
    uvicorn.run(app, host="0.0.0.0", port=8001)
