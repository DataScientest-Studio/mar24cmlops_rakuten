
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from prediction import predict_with_unified_interface
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

# A supprimer à la fin
@app.post('/predictionT')
async def test_image4(designation : str =None, imageid : int = None, productid : int = None, directory : str = 'data/preprocessed/image_train', new_image : str = None, file : UploadFile | None = None):
    img_context=None
    if file is not None :
        img_context= await file.read()
    return {'prediction' : predict_with_unified_interface(designation=designation, imageid=imageid,productid=productid,directory=directory,new_image=new_image,file=img_context)}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)
