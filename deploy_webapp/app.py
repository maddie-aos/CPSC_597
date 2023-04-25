from flask import Flask, render_template, request
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import model_from_json
import h5py
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import rasterio
from osgeo import gdal, gdal_array
from shapely.geometry import Point
import geopandas as gpd

app = Flask(__name__)

#loading keras models: present
def load_cit_sor_model():
    json_file = open('results/fish/Citharichthys_sordidus/Citharichthys_sordidus_model.json')
    loaded_model_csor = json_file.read()
    json_file.close()
    cit_sor_model = model_from_json(loaded_model_csor)
    cit_sor_model.load_weights('results/fish/Citharichthys_sordidus/Citharichthys_sordidus_model.h5')
    cit_sor_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

 
def load_eng_mor_model():
    json_file = open('results/fish/Engraulis_mordax/Engraulis_mordax_model.json')
    loaded_model_emor = json_file.read()
    json_file.close()
    eng_mor_model = model_from_json(loaded_model_emor)
    eng_mor_model.load_weights('results/fish/Engraulis_mordax/Engraulis_mordax_model.h5')
    eng_mor_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


def load_par_cal_model():
    json_file = open('results/fish/Paralichthys_californicus/Paralichthys_californicus_model.json')
    loaded_model_pcal = json_file.read()
    json_file.close()
    par_cal_model = model_from_json(loaded_model_pcal)
    par_cal_model.load_weights('results/fish/Paralichthys_californicus/Paralichthys_californicus_model.h5')
    par_cal_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


def load_sco_jap_model():
    json_file = open('results/fish/Scomber_japonicus/Scomber_japonicus_model.json')
    loaded_model_sjap = json_file.read()
    json_file.close()
    sco_jap_model = model_from_json(loaded_model_sjap)
    sco_jap_model.load_weights('results/fish/Scomber_japonicus/Scomber_japonicus_model.h5')
    sco_jap_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_thu_ala_model():
    json_file = open('results/fish/Thunnus_alalunga/Thunnus_alalunga_model.json')
    loaded_model_tala = json_file.read()
    json_file.close()
    thu_ala_model = model_from_json(loaded_model_tala)
    thu_ala_model.load_weights('results/fish/Thunnus_alalunga/Thunnus_alalunga_model.h5')
    thu_ala_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_xip_gla_model():
    json_file = open('results/fish/Xiphias_gladius/Xiphias_gladius_model.json')
    loaded_model_xgla = json_file.read()
    json_file.close()
    xip_gla_model = model_from_json(loaded_model_xgla)
    xip_gla_model.load_weights('results/fish/Xiphias_gladius/Xiphias_gladius_model.h5')
    xip_gla_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


#loading keras models: future
def load_cit_sor_model_future():
    json_file = open('results/fish_future/Citharichthys_sordidus/Citharichthys_sordidus_future_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global cit_sor_model_future 
    cit_sor_model_future  = model_from_json(loaded_model_json)
   

    cit_sor_model_future.load_weights('results/fish_future/Citharichthys_sordidus/Citharichthys_sordidus_future_model.h5')
    cit_sor_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_eng_mor_model_future():
    json_file = open('results/fish_future/Engraulis_mordax/Engraulis_mordax_future_model.json')
    loaded_model_emorf = json_file.read()
    json_file.close()
    eng_mor_model_future = model_from_json(loaded_model_emorf)
    eng_mor_model_future.load_weights('results/fish_future/Engraulis_mordax/Engraulis_mordax_future_model.h5')
    eng_mor_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_par_cal_model_future():
    json_file = open('results/fish_future/Paralichthys_californicus/Paralichthys_californicus_future_model.json')
    loaded_model_pcalf = json_file.read()
    json_file.close()
    par_cal_model_future = model_from_json(loaded_model_pcalf)
    par_cal_model_future.load_weights('results/fish_future/Paralichthys_californicus/Paralichthys_californicus_future_model.h5')
    par_cal_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_sco_jap_model_future():
    json_file = open('results/fish_future/Scomber_japonicus/Scomber_japonicus_future_model.json')
    loaded_model_sjapf = json_file.read()
    json_file.close()
    sco_jap_model_future = model_from_json(loaded_model_sjapf)
    sco_jap_model_future.load_weights('results/fish_future/Scomber_japonicus/Scomber_japonicus_future_model.h5')
    sco_jap_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_thu_ala_model_future():
    json_file = open('results/fish_future/Thunnus_alalunga/Thunnus_alalunga_future_model.json')
    loaded_model_talaf = json_file.read()
    json_file.close()
    thu_ala_model_future = model_from_json(loaded_model_talaf)
    thu_ala_model_future.load_weights('results/fish_future/Thunnus_alalunga/Thunnus_alalunga_future_model.h5')
    thu_ala_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_xip_gla_model_future():
    json_file = open('results/fish_future/Xiphias_gladius/Xiphias_gladius_future_model.json')
    loaded_model_xglaf = json_file.read()
    json_file.close()
    xip_gla_model_future = model_from_json(loaded_model_xglaf)
    xip_gla_model_future.load_weights('results/fish_future/Xiphias_gladius/Xiphias_gladius_future_model.h5')
    xip_gla_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])





#Setting the main pages
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/present")
def present():
    return render_template("present.html")

@app.route("/future")
def future():
    return render_template("future.html")


#Selecting Species in the Present
@app.route("/cit_sor_pres")
def cit_sor_pres():
    return render_template("cit_sor_pres.html")

@app.route("/eng_mor_pres")
def eng_mor_pres():
    return render_template("eng_mor_pres.html")

@app.route("/par_cal_pres")
def par_cal_pres():
    return render_template("par_cal_pres.html")

@app.route("/sco_jap_pres")
def sco_jap_pres():
    return render_template("sco_jap_pres.html")

@app.route("/thu_ala_pres")
def thu_ala_pres():
    return render_template("thu_ala_pres.html")

@app.route("/xip_gla_pres")
def xip_gla_pres():
    return render_template("xip_gla_pres.html")


#Selecting Species in the Future
@app.route("/cit_sor_fut")
def cit_sor_fut():
    return render_template("cit_sor_fut.html")

@app.route("/eng_mor_fut")
def eng_mor_fut():
    return render_template("eng_mor_fut.html")

@app.route("/par_cal_fut")
def par_cal_fut():
    return render_template("par_cal_fut.html")

@app.route("/sco_jap_fut")
def sco_jap_fut():
    return render_template("sco_jap_fut.html")

@app.route("/thu_ala_fut")
def thu_ala_fut():
    return render_template("thu_ala_fut.html")

@app.route("/xip_gla_fut")
def xip_gla_fut():
    return render_template("xip_gla_fut.html")


#Present Distribution Pages
@app.route("/cit_sor_dist")
def cit_sor_dist():
    return render_template("cit_sor_dist.html")

@app.route("/eng_mor_dist")
def eng_mor_dist():
    return render_template("eng_mor_dist.html")

@app.route("/par_cal_dist")
def par_cal_dist():
    return render_template("par_cal_dist.html")

@app.route("/sco_jap_dist")
def sco_jap_dist():
    return render_template("sco_jap_dist.html")

@app.route("/thu_ala_dist")
def thu_ala_dist():
    return render_template("thu_ala_dist.html")

@app.route("/xip_gla_dist")
def xip_gla_dist():
    return render_template("xip_gla_dist.html")


#Future Distributions Pages
@app.route("/cit_sor_fut_dist")
def cit_sor_fut__dist():
    return render_template("cit_sor_fut_dist.html")

@app.route("/eng_mor_fut_dist")
def eng_mor_fut_dist():
    return render_template("eng_mor_fut_dist.html")

@app.route("/par_cal_fut_dist")
def par_cal_fut_dist():
    return render_template("par_cal_fut_dist.html")

@app.route("/sco_jap_fut_dist")
def sco_jap_fut_dist():
    return render_template("sco_jap_fut_dist.html")

@app.route("/thu_ala_fut_dist")
def thu_ala_fut_dist():
    return render_template("thu_ala_fut_dist.html")

@app.route("/xip_gla_fut_dist")
def xip_gla_fut_dist():
    return render_template("xip_gla_fut_dist.html")


#Predictions: Present
@app.route("/cit_sor_pred")
def cit_sor_pred():
    return render_template("cit_sor_pred.html")

@app.route("/eng_mor_pred")
def eng_mor_pred():
    return render_template("eng_mor_pred.html")

@app.route("/par_cal_pred")
def par_cal_pred():
    return render_template("par_cal_pred.html")

@app.route("/sco_jap_pred")
def sco_jap_pred():
    return render_template("sco_jap_pred.html")

@app.route("/thu_ala_pred")
def thu_ala_pred():
    return render_template("thu_ala_pred.html")

@app.route("/xip_gla_pred")
def xip_gla_pred():
    return render_template("xip_gla_pred.html")


#Predictions: Future

@app.route("/cit_sor_fut_pred")
def cit_sor_fut_pred():
    return render_template("cit_sor_fut_pred.html")

@app.route("/cit_sor_fut_pred", methods = ['POST'])
def predict():
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    #reading raster and extracting values 
    inRas=gdal.Open('data/modified_data/stacked_bio_oracle_future/bio_oracle_future_stacked.tif')
    myarray=inRas.ReadAsArray()

    len_pd=np.arange(len(df))
    lon=df["deci_lon"]
    lat=df["deci_lat"]
    lon=lon.values
    lat=lat.values
    
    
    row=[]
    col=[]


    src=rasterio.open('data/modified_data/stacked_bio_oracle_future/bio_oracle_future_stacked.tif', crs= 'espg: 4326')
    for i in len_pd:
        row_n, col_n = src.index(lon[i], lat[i])# spatial --> image coordinates
        row.append(row_n)
        col.append(col_n)
    
    mean_std=pd.read_csv('data/modified_data/stacked_bio_oracle_future/future_env_bio_mean_std.txt',sep="\t")
    mean_std=mean_std.to_numpy()

    X=[]
    for j in range(0,9):
        print(j)
        band=myarray[j]
        x=[]
        
        for i in range(0,len(row)):
            value= band[row[i],col[i]]
            if value <-1000:
                x.append(value)
            else:
                value = ((value - mean_std.item((j,1))) / mean_std.item((j,2))) # scale values
                x.append(value)
        X.append(x)

    X.append(row)
    X.append(col)

    X =np.array([np.array(xi) for xi in X])
    
    df=pd.DataFrame(X)
    df=df.T
    
    #drop any rows with no-data values
    df=df.dropna(axis=0, how='any')
    input_X=df.loc[:,0:8]
    
    row=df[9]
    col=df[10]
    
    row_col=pd.DataFrame({"row":row,"col":col})
   
    input_X=input_X.values
    
    #convert rows and col indices back to array
    row=row.values
    col=col.values
    
    prediction_array=np.save('deploy_webapp/predictions/prediction_array.npy',input_X)
    prediction_pandas=row_col.to_csv('deploy_webapp/predictions/prediction_row_col.csv')

    #predicting outcome
    input_X=np.load('deploy_webapp/predictions/prediction_array.npy')
    df=pd.DataFrame(input_X)

    #create copy of band to later subset values in
    new_band=myarray[1].copy()
    new_band.shape

    new_values = cit_sor_model_future.predict(x=input_X,verbose=0) ###predict output value

    ##take the prob. of presence (new_value.item((0,1))) and put into numpy array
    new_band_values=[]
   
    for i in new_values:
        new_value=i[1]
        new_band_values.append(new_value)
    new_band_values=np.array(new_band_values)

    return flask.jsonify(new_band_values)

@app.route("/eng_mor_fut_pred")
def eng_mor_fut_pred():
    return render_template("eng_mor_fut_pred.html")

@app.route("/par_cal_fut_pred")
def par_cal__fut_pred():
    return render_template("par_cal_fut_pred.html")

@app.route("/sco_jap_fut_pred")
def sco_jap_fut_pred():
    return render_template("sco_jap_fut_pred.html")

@app.route("/thu_ala_fut_pred")
def thu_ala_fut_pred():
    return render_template("thu_ala_fut_pred.html")

@app.route("/xip_gla_fut_pred")
def xip_gla_fut_pred():
    return render_template("xip_gla_fut_pred.html")


#Loading machine learning models

if __name__ == "__main__":
    app.run(debug=True)
    
