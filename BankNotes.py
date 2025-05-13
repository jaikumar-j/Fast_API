from pydantic import BaseModel
# 2Class describe banknote

class BankNote(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float