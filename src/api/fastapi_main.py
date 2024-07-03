from datetime import timedelta
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import duckdb
import os
from utils.make_db import download_initial_db
from utils.security import create_access_token, verify_password
from dotenv import load_dotenv
import jwt
from utils.resolve_path import resolve_path
from prediction import predict_with_unified_interface
from datetime import datetime
from utils.s3_utils import create_s3_conn_from_creds, download_from_s3, upload_to_s3
from utils.make_db import upload_db, download_initial_db
import boto3 
import pandas as pd 
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
import re

import uvicorn

s3_client=None
db_conn=None
prd_categories=dict()

# Load environment variables from .env file
env_path = resolve_path('.env/.env.development')
load_dotenv(env_path)

# Convert paths to absolute paths
aws_config_path = resolve_path(os.environ['AWS_CONFIG_PATH'])
duckdb_path = os.path.join(resolve_path(os.environ['DATA_PATH']), os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
encrypted_file_path = os.path.join(resolve_path(os.environ['AWS_CONFIG_FOLDER']), '.encrypted')

# Check if the DuckDB database file exists locally, if not, download it from S3
if not os.path.isfile(duckdb_path):
    print('No Database Found locally')  
    # Since no database found for the API, download the initial database from S3
    download_initial_db(aws_config_path, duckdb_path)
    print('Database Sucessfully Downloaded')
    
#duckdb_path = os.path.join(os.environ['DATA_PATH'], os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']

# Download database for the mapping of the results
db_conn = duckdb.connect(database=duckdb_path, read_only=False)
s3_client = create_s3_conn_from_creds(aws_config_path)

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
    cursor = db_conn.execute(f"SELECT hashed_password FROM dim_user WHERE username = '{username}'")
    result = cursor.fetchone()
    if not result:
        return False
    
    hashed_password = result[0]
    return verify_password(password, hashed_password)

# Define OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FastAPI app
app = FastAPI()

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
    cursor = db_conn.execute(f"SELECT {columns_str} FROM fact_listings WHERE listing_id = {listing_id}")
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
    cursor = db_conn.execute(f"SELECT user FROM fact_listings WHERE listing_id = {listing_id}")
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")
    if result[0] != current_user["username"]:
        raise HTTPException(status_code=403, detail="This user is not the owner of this listing_id")
    # Logic to delete listing
    db_conn.execute(f"DELETE FROM fact_listings WHERE listing_id = {listing_id}")
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
    
    listing_id_added = db_conn.execute(sql).fetchall()[0][0]

    return {"message": f"Listing {listing_id_added} added successfully"}

def mise_en_forme(text: str):
    text=re.sub("'"," ",text)
    return "'"+text+"'"

@app.post("/listing_submit")
async def listing_submit(designation : str = None, description : str = None, imageid : int = None, productid : int = None , directory : str = 'image_train', new_image : str = None, file : UploadFile = File(None), current_user : dict = Depends(get_current_user)):
    # Verify if object already recorded if imageid and productid provided
    reponse=None
    if imageid is not None and productid is not None:
        reponse=db_conn.sql(f"SELECT * FROM fact_listings WHERE imageid={imageid} AND productid={productid};").fetchone()
    if reponse is not None :
        return {"message" : "Objet déjà enregistré."}

    # Image handle
    img=None
    if new_image is not None :
        img=load_img(new_image,target_size=(224,224,3))
    if file is not None:
        img_context= await file.read()
        img=load_img(BytesIO(img_context),target_size=(224,224,3))
    if img is None :
        return {'Message' : 'pas d\'image disponible'}
    
    # Formatting of the data (text and datetime)
    waiting_date=mise_en_forme(str(datetime.now()))
    description=mise_en_forme(description)
    designation=mise_en_forme(designation)
    utilisateur=mise_en_forme(current_user['username'])
    
    # Automatic affectation of new imageid and productid    
    new_imageid, new_productid, new_listingid=db_conn.sql('SELECT MAX(imageid)+1,MAX(productid)+1,MAX(listing_id)+1 from fact_listings;').fetchone()
    print(f"INSERT INTO fact_listings (listing_id,imageid,productid,designation,description,user,waiting_datetime) VALUES ({new_listingid},{new_imageid},{new_productid},{designation},{description},{utilisateur},{waiting_date});") 
    db_conn.sql(f"INSERT INTO fact_listings (listing_id,imageid,productid,designation,description,user,waiting_datetime) VALUES ({new_listingid},{new_imageid},{new_productid},{designation},{description},{utilisateur},{waiting_date});")        

    # save the image locally
    save_img(resolve_path(f"data/temporary_images/image_{imageid}_product_{productid}.jpg"),img)
    prdtypecode=predict_with_unified_interface(s3_client=s3_client, designation=designation, imageid=new_imageid,productid=new_productid,directory=directory,new_image=new_image,file=img_context)[0] 

    # Update of the local database table
    db_conn.sql(f"UPDATE fact_listings SET model_prdtypecode={prdtypecode}, status='waiting' WHERE listing_id={new_listingid};")
    # upload_to_s3(s3_client,"data/rakuten_db.duckdb",'db/rakuten_db.duckdb')
    # upload_to_s3(s3_client,f"data/temporary_images/image_{imageid}_product_{productid}.jpg",f"image_train/image_{imageid}_product_{productid}.jpg")
    return {'Identifiant de l\'annonce' : new_listingid}
        
@app.post("/listing_validate")
async def listing_validate(listingid : int = None, prdtypecode : int = None, current_user: dict = Depends(get_current_user)): 
    """
    10      :   "Livres et oeuvres sérieuses"   
    40      :   "Import de CD et jeux vidéos"   
    50      :   "Hardware jeu vidéo et accessoires informatique"    
    60      :   "Console et jeu vidéo rétro"    
    1140    :   "Figurines et autres goodies"    
    1160    :   "Cartes (magic, pokemon...)"    
    1180    :   "JDR"       
    1280    :   "Jouet neuf"    
    1281    :   "Jouet de seconde main"    
    1300    :   "Modélisme, drônes"    
    1301    :   "Divers"  
    1302    :   "Accessoires associés aux loisirs en extérieur (randonnée, pêche...)"     
    1320    :   "Accessoires pour bébé"   
    1560    :   "Mobilier d'intérieur divers"   
    1920    :   "Coussin, taie, oreiller, rideau..."    
    1940    :   "Alimentation, nourriture"  
    2060    :   "Décoration intérieur"  
    2220    :   "Accessoires pour animaux"                  
    2280    :   "Journaux,magazines,revues" 
    2403    :   "Livres de seconde main"    
    2462    :   "Console et jeu vidéo de seconde main"  
    2522    :   "Papetterie"       
    2582    :   "Abri de jardin, serre, barrière, banc..."    
    2583    :   "Matériel piscine"  
    2585    :   "Ensemble d'outils associés à l'extérieur"      
    2705    :   "Livres neufs populaires"   
    2905    :   "Téléchargement (jeu vidéo...)" 
    """
    requete=db_conn.sql(f"SELECT user FROM fact_listings WHERE listing_id={listingid};").fetchone()
    # Error message sent if wrong listingid
    if not requete :
        return {'Message' : 'Mauvais identifiant d\'annonce'}
    # Error message send if user not authorized 
    if current_user['username']!=requete[0] and current_user['access_rights']!='administrator':
        return {'Message' : 'Mauvais utilisateur pour cette annonce'}
    # Update of the local duckdb table
    validate_date=mise_en_forme(str(datetime.now()))
    db_conn.sql(f"UPDATE fact_listings SET user_prdtypecode={prdtypecode}, validate_datetime={validate_date}, status='validate' WHERE listing_id={listingid};")
    # upload_to_s3(s3_client,"/data/rakuten_db.duckdb",'db/rakuten_db.duckdb')
    return {'Message':'Validation'}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)