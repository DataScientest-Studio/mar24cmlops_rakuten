
from fastapi import FastAPI, HTTPException, Depends, status
from prediction import predict_with_unified_interface
from fastapi import UploadFile, File
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def get_index():
    return {'data': 'hello world '+os.getcwd()}

@app.post('/predict')
async def prediction(designation : str =None, imageid : int = None, productid : int = None, directory : str = 'data/preprocessed/image_train', new_image : str = None, file : UploadFile | None = None):
    img_context=None
    if file is not None :
        img_context= await file.read()
    return {'prediction' : predict_with_unified_interface(designation=designation, imageid=imageid,productid=productid,directory=directory,new_image=new_image,file=img_context)}



if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)
