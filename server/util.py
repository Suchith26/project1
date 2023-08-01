import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def est_price(location ,sqft , bhk , bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index=-1

    x= np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bhk
    x[2]=bath
    if loc_index >=0:
        x[loc_index]=1

    return round(__model.predict([x])[0],2)
def locations():
    return __locations

def load_saved():
    print("loading saved artifacts..")
    global __data_columns
    global __locations

    with open("./data/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("./data/banglore_home_price.pickle",'rb')as f:
        __model= pickle.load(f)
    print("loading is done")
if __name__ == '__main__':
    load_saved()
    print(locations())
