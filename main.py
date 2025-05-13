from fastapi import FastAPI, Path,Form,File, UploadFile, HTTPException
from typing import Union
from enum import Enum
from pydantic import BaseModel
import uvicorn 
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app=FastAPI()
csv_file_path = "path_your _csv"
data=pd.read_csv(csv_file_path)
X = data[['variance', 'skewness', 'curtosis', 'entropy']]
y = data['class']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

class schema1(BaseModel):
    name:str
    Class:str
    roll_no:str

class ChoiceNames (str,Enum):
    one ="Jai"
    two="Prince"
    three="Syed"
    four="Pulkit"
    five="Subham"
    six="Mayank"
    seven="Rachna"
    eight="Ranu"
    nine="Atul jain"
    ten="Anurag"

app = FastAPI()

@app.get("/hello")
async def root():
    return {"message": "Hello World"}

@app.get("/jai")
async def root():
    return {"message": "how can i help you"}

@app.get("/item/{item_id}")
def path_func(item_id: str):
    var_name={"path variable":item_id}
    return {"item_id": item_id}

@app.get("/query/")
def query_func(name: str, roll_no:int):
    var_name= {"name":name, "roll_no":roll_no}
    return(var_name )

#union
@app.get("/query")
def query_func(name: str, roll_no:Union[int,None]=None):
    var_name= {"name":name, "roll_no":roll_no}
    return(var_name )

#choice filed
@app.get("/models/{model_name}")
async def get_model(model_name: ChoiceNames):
    if model_name == ChoiceNames.one:
        return {"model_name": model_name, "message": "calling One"}

    if model_name == ChoiceNames.two:
        return {"model_name": model_name, "message": "calling Two"}
    
    if model_name == ChoiceNames.three:
        return {"model_name": model_name, "message": "calling three"}
    
    if model_name == ChoiceNames.four:
        return {"model_name": model_name, "message": "calling four"}
    
    if model_name == ChoiceNames.five:
        return {"model_name": model_name, "message": "calling five"}

    if model_name == ChoiceNames.six:
        return {"model_name": model_name, "message": "calling six"}

    if model_name == ChoiceNames.seven:
        return {"model_name": model_name, "message": "calling seven"}

    if model_name == ChoiceNames.eight:
       return {"model_name": model_name, "message": "calling eight"}
    
    if model_name == ChoiceNames.nine:
        return {"model_name": model_name, "message": "calling nine"}
    
    return {"model_name": model_name, "message": "calling ten"}

# request body
@app.get("/items/")
async def create_item(item:schema1):
    return item

#form data
@app.post("/form/data")
async def form_data(username :str=Form(),password: str = Form()):
    return({"username":username,"password":password})

# upload file
@app.post("/file/size")
async def file_bytes_len(file : bytes=File()):
    return({"file":len(file)})

@app.post("/file/info")
async def file_upload(file : UploadFile):
    return({"file":file})

# error handling
@app.post("/error/handle")
async def handle_error(item:int):
    if item ==5 :
        return HTTPException(status_code=400,detail="item is not equal , try another value")
    return({"value":item})

# deploy the Ml model data using fastapi
@app.post("/predict")
def predict_banknote(data:BankNote):
    input_data = [[data.variance, data.skewness, data.curtosis, data.entropy]]
    prediction = classifier.predict(input_data)[0]
    if prediction==1:
        prediction="fake note"
    else:
        prediction="Its bank note"
    return{
        'prediction':prediction
    }
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)


