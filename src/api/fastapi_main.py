from fastapi import FastAPI, Depends, status, HTTPException, Body, Header, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from typing import Optional
import jwt
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import shutil
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
import os
from PIL import Image
#import predict

api=FastAPI(title="API Rakuten",description="API simulation Rakuten",version="1.0.1")

pwd_context=CryptContext(schemes=["bcrypt"],deprecated="auto")
oauth_scheme=OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY="JslTVIpATA"
ALGORITHM="HS256"

# En attendant d'implémenter la database # WRAC:Desdichado V1JBQzpEZXNkaWNoYWRv
users_db={'WRAC':{"username":"WRAC",
                  "nom":"",
                  "prénom":"",
                  "société":"",
                  "adresse électronique":"",
                  "type_user":"administrator",
                  "hashed passwd":pwd_context.hash("Desdichado")}}


class Utilisateur(BaseModel):
    username : str
    nom : Optional[str]
    prenom : Optional[str]
    societe : Optional[str]
    email : Optional[str]
    passwd : str

class Token(BaseModel):
    access_token: str
    token_type: str

def create_access_token(data:dict, expires_delta: timedelta = None):
    a_encoder=data.copy()
    if expires_delta:
        expire=datetime.utcnow()+expires_delta
    else:
        expire=datetime.utcnow()+timedelta(minutes=15)
    a_encoder.update({'exp':expire})
    encoded_jwt=jwt.encode(a_encoder,SECRET_KEY,algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token : str=Depends(oauth_scheme)):
    try:
        payload=jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username : str = payload.get('sub')
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Pas d'username",
                                headers={'WWW-Authenticate':'Bearer'})
    except :
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Problème d'identifiction (mot de passe?)",
                            headers={"WWW-Authenticate":"Bearer"})
    # Partie à modifier dès la BBD dim_user implémentée
    user=users_db.get(username,None)
    #
    if user is None :
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Utilisateur non enregistré",
                            headers={"WWW-Authenticate":"Bearer"})
    return user


@api.get('/hello')
def bonjour():
    return {'greetings' : 'bonjour'}

@api.get('/subscribe')
async def enregistrement(utilisateur : Utilisateur):
    return_code=200
    user_id='0'
    #con=duck.connect('db')
    try:
        if utilisateur.username is None:
            raise Exception("Pas d'username")
        if users_db.get(utilisateur.username) is None:
            raise Exception("Username déjà utilisé")
        ## à remplacer avec duckdb:
        #if not con.sql(f"SELECT username FROM dim_user WHERE dim_user.username=={utilisateur.username};""):
        #   raise Exception('Déjà présent')
        if utilisateur.passwd is None:
            raise Exception("Pas de mot de passe")
        hashed = pwd_context.hash(b"utilisateur.passwd")
        ## Partie à remplacer avec duckdb, prévoir création
        users_db.update({utilisateur.username : utilisateur})
        ##
        #con=duckdb.connect('dim_user')
        #con.sql(f"INSERT  INTO dim_user VALUES ({utilisateur.username},{utilsateur.nom},{utilisateur.prenom},
        # {utilisateur.societe},{utilisateur.email}, {hashed});"")
    except:
        return_code=400
    finally:
        con.close()
        with open('user_log',"a") as file:
            file.write(",".join([datetime.utcnow().strftime("%m/%d/%Y-%H:%M:%S"),'/subscribe',utilisateur.username,str(return_code),user_id,'user']))

@api.post('/auth',response_model=Token)
async def login(form_data : OAuth2PasswordRequestForm = Depends()):
    return_code=200
    user_id='0'
    access_token=None
    #con=duckdb.connect('db')
    try:
        ## A remplacer dès duckdb
        user=users_db.get(form_data.username)
        ## Alternative duckdb
        #user=con.execute(f"SELECT username, type_user, hashed FROM dim_user WHERE username=={form_data.username};").fetchone()
        #hashed_passwd=user[2]
        #user={'username':user[0],'type_user':user[1]}
        hashed_passwd=user.get("hashed passwd")
        if not user or not pwd_context.verify(form_data.password,hashed_passwd):
            raise HTTPException(status_code=400,
                            detail="username ou password incorrect")
        access_token=create_access_token(data={'sub':form_data.username})
        return {'access_token' : access_token, 'token_type' : 'bearer'}
    except:
        return_code=400
    finally:
        with open('user_log',"a") as file:
            file.write(",".join([datetime.utcnow().strftime("%m/%d/%Y-%H:%M:%S"),'/subscribe',user["username"],str(return_code),user_id,user["type_user"]])+'\n')

# A supprimer à la fin
@api.post('/image')
async def test_image(designation : str, description : str, file : UploadFile = File(...), tok : dict=Depends(get_current_user)):
    with open('A_'+file.filename,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    return {'filename':file.filename,'dict':tok,'designation':designation,'description':description}

# A supprimer à la fin
@api.get('/test_temp')
async def test_temp(tok : dict = Depends(get_current_user)):
    return tok

@api.post('/listing_submit',response_model=Token, name="soumission des produits")
async def l_submit(designation : str, description : str, file : UploadFile = File(...), user : dict = Depends(get_current_user)):
    conn=duckdb.connect('db')
    utilisateur=user['username']
    # Placer un verrou sur l'accès à la database ??????
    # Possibilité de faire toutes ces requêtes en une fois ?
    image_id=int(conn.sql('SELECT MAX(imageid) FROM fac_listings').fetchone()[0])+1
    productid=int(conn.sql('SELECT MAX(productid) FROM fact_listings').fetchone()[0])+1
    listings_id=int(conn.sql('SELECT MAX(listing_id) FROM fact_listings').fetchone()[0])+1
    #conn.sql(f'INSERT INTO fac_listings VALUES ({listing_id},{designation},{description},{imageid},{productid},"","",{utilisateur},
    #          "waiting",{datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S")})')
    # Libérer le verrou ???
    # chemin à spécifier
    filename=f"init_image_{image_id}_product_{productid}.jpg"
    with open(filename,'wb') as buffer :
        shutil.copyfileobj(file.file,buffer)
    image=Image.open(filename)
    image=image.resize((500,500),Image.ANTIALIAS)  # Je présuppose que l'image est déjà en couleur sinon il faudra rajouter une dimension
    # chemin à spécifier
    mpimg.imwrite(filename[5:],image)
    ### Prediction
    prediction=  ###### Appel à predict.py
    ################### A terminer
    return=200
    with open('product_log','a') as file:
        file.write("".join([datetime.utcnow().strftime("%m/%d/%Y-%H:%M:%S"),'/listing_submit',str(return_code),utilisateur,str(listing_id),
        ])+'\n') ###############################
    return {'listing_id':listings_id,'model_prdtypecode':prediction}

@api.post('/listing_validate',name="vaidation du produit")
async def validation(listing_id : str, categorie : str, tok : dict = Depends(get_current_type):
    utilisateur=user['username']
    conn=duckdb.connect('db')
    ptc = int(conn.sql(f'SELECT prdtypecode from fac_listings WHERE category={categorie};').fetchone()[0])
    conn.sql(f'UPDATE dim_prdtypecode SET user_prdtypcode={ptc},statut="validate"  WHERE listing_id={listing_id}';)
    return_code=200
    with open('product_log','a') as file:
        file.write("".join([datetime.utcnow().strftime("%m/%d/%Y-%H:%M:%S"),'listing_validate',str(return_code),utilisateur,str(listing_id),
        ])+'\n') ########################
    return "Youpi"

@api.post('/log/{log_name : str}')  # Compléter avec des try : except:
async def journal(log_name : str,tok : dict = Depends(get_current_user)):
    utilisateur=tok['username']
    conn=duckdb.connect('db')
    droit=conn.sql(f'SELECT type_user FROM dim_user WHERE username={utiisateur};').fetchone()[0]
    if droit!='administrator':
        raise Exception('Accès non autorisé')
    if log_name not in ['user_log','product_log','monitoring_log']:
        raise Exception("Mauvais journal")
    texte=""
    with open(log_name,'r') as file:
        texte="".join(file.readlines())
    return {log_name : texte}


#@api.get('/model_retrain')
async def retrain(tok : dict = Depends(get_current_user)):
    conn=duckdb.connect('db')
    df=duckdb.sql('SELECT designation,description,imageid,productid FROM fac_listings;').df()
    #### lancement ré-entraînement : trouver un moyen de le renommer
    with open('monitoring_log','a') as file:
        file.write("".join([]+'\n'))   ################### A compléter

@api.get('/model_compare')
async def compare(model1 : str, model2 : str, tok : dict = Depends(get_current_user)):
    """
    1_On s'assure que l'utilisateur est un administrateur.
    2_On effectue une prédiction sur chacun des modèles spécifiés en entrée depuis les entrées récentes(numéro issu des logs).
    3_On enregistre les scores sur le journal monitoring_log.
    """
    conn=duckdb.connect('db')
    # Pour l'instant, les données récentes sont celles non issues du X_train.
    df=duckdb.sql('SELECT designation, description, imageid, productid FROM fac_listings WHERE NOT username='0';).df()
    ######
    ##### Prédiction sur les deux modèles en cours et production des données score1 et score2
    #####
    with open('monitoring_log','a') as file :
        file.write("".join([]+'\n'))  ###################### A compléter

@api.get('/model_unstage')
async def unstage(model : str,tok : dict = Depends(get_current_user)):
    os.remove(os.path.join())
    # Vérifier l'existence du modèle avant effacement
    with open('monitoring_log','a') as file :
        file.write("".join([]+'\n'))  ###################### A compléter


## Optionnel

#@api.put('/listing_update)
#@api.get('/listing_read')
#@api.delete('/listing_delete')
#@api.get('/exit')
#@api.get('/get_listing')
#@api.post('/predict_typecode')

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8001)