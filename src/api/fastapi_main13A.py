
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from prediction12 import text_predict, image_predict, predict, image_predict_object, predictionT
from fastapi import UploadFile, File
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import shutil
from typing import Annotated, Union

class testo(BaseModel):
    text : str

class picture(BaseModel):
    imageid : int
    productid : int
    directory : Optional[str] = None
    
# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def get_index():
    return {'data': 'hello world '+os.getcwd()}


@app.post("/prediction_text")
async def prediction_text(designation : str):
    """
    Endpoint test for verify the functionality of the module prediction.py
    Args :
        designation.
        current_user (dict): Dictionary containing current user information.
    Returns:
        the predicted prdtypecode
    """
    #return {'endpoint prediction_text' : designation}
    return {"Prediction text" : text_predict(designation)[0]}

@app.post("/prediction_image")
async def prediction_image(imageid : int, productid : int, directory : str = None):
    """
    Endpoint test for verify the functionality of the module prediction.py
    Args :
        imageid and productid : to check an already recorded image.
        current_user (dict): Dictionary containing current user information.
    Returns:
        the predicted prdtypecode
    """
    #productid=000000
    #return {'endpoint prediction_image' : f"image_{str(imageid)}_product_{str(productid)}.jpg"}
    return {"Prediction image" : image_predict(imageid=imageid, productid=productid, directory=directory)[0]}

@app.post("/prediction_image_object")
async def prediction_image_object(image : UploadFile=File(...)):
    """
    Endpoint test for verify the functionality of the module prediction.py
    Args :
        imageid and productid : to check an already recorded image.
        current_user (dict): Dictionary containing current user information.
    Returns:
        the predicted prdtypecode
    """
    image_context= await image.read()
    #picture=Image.open(BytesIO(image_context))
    picture = load_img(BytesIO(image_context))
    return {"Prediction image" : image_predict_object(picture)[0]}


# A supprimer à la fin
@app.post('/image1')
async def test_image1(file : UploadFile = File(...)):
    with open('A_test.jpg',"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    return {'oups':'ok'}

# A supprimer à la fin
@app.post('/image2')
async def test_image2(file : UploadFile = File(...)):
    img_context=await file.read()
    img=Image.open(BytesIO(img_context))
    img.save('B_test.jpg')
    return {'oups':'ok'}

# A supprimer à la fin
@app.post('/image3')
async def test_image3(file : UploadFile = File(...)):
    img_context=await file.read()
    img=load_img(BytesIO(img_context),target_size=(224,224,3))
    #img.save('C_test.jpg')
    return {'Prediction image 3':image_predict_object(img)[0]}

# A supprimer à la fin
#@app.post('/image4')
#async def test_image4(designation : str =None, imageid : int = None, productid : int = None, directory : str = None, nouveau : str = None, file : UploadFile | None = None):
#    img=None
#    if file is not None :
#        #return {'Prediction image 4':'coucou'}
#        img_context=await file.read()
#        img=load_img(BytesIO(img_context),target_size=(224,224,3))
#        #img.save('C_test.jpg')
#        return {'Prediction image 4':image_predict_object(img)[0]}
#    if imageid is not None :
#        if productid is not None :
#            if directory is None :
#                directory="."
#            return {'Prediction image 4': image_predict(imageid=imageid,productid=productid,directory=directory)[0]}
#    if nouveau is not None:
#        return {'Prediction image 4' : image_predict(nouveau=nouveau)[0]}
#    return {'Prediction image 4': 'Au revoir'}

# A supprimer à la fin
@app.post('/predictionT')
async def test_image4(designation : str =None, imageid : int = None, productid : int = None, directory : str = 'data/preprocessed/image_train', new_image : str = None, file : UploadFile | None = None):
    img_context=None
    if file is not None :
        img_context= await file.read()
    return {'prediction' : predictionT(designation=designation, imageid=imageid,productid=productid,directory=directory,new_image=new_image,file=img_context)}

@app.post("/prediction")
async def prediction_test(designation : str, imageid : int, productid : int, directory : str = None):
    """
    Endpoint test for verify the functionality of the module prediction.py
    Args :
        listing (Listing): Listing information to be added.
        current_user (dict): Dictionary containing current user information.
    Returns:
        the predicted prdtypecode
    """
    return {"Predicted prdtypecode" : predict(entry=designation, imageid=imageid, productid=productid, directory=directory)}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)
