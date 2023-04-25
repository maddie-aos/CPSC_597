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

app = Flask(__name__)
#getting test value data for present
eval_pres = pd.read_csv('results/DNN_performance/DNN_eval.txt', sep='\t', header=0)

#getting test value data for future
eval_fut= pd.read_csv('results/DNN_performance/DNN_eval_future.txt', sep='\t', header=0)

#loading keras models: present
cit_sor_model = load_model('deploy_webapp/saved_models/Citharichthys_sordidus.h5')
eng_mor_model = load_model('deploy_webapp/saved_models/Engraulis_mordax.h5')
par_cal_model = load_model('deploy_webapp/saved_models/Paralichthys_californicus.h5')
sco_jap_model = load_model('deploy_webapp/saved_models/Paralichthys_californicus.h5')
thu_ala_model = load_model('deploy_webapp/saved_models/Thunnus_alalunga.h5')
xip_gla_model = load_model('deploy_webapp/saved_models/Xiphias_gladius.h5')

#loading keras models: future
cit_sor_model_future = load_model('deploy_webapp/saved_models/Citharichthys_sordidus_future.h5')
eng_mor_model_future = load_model('deploy_webapp/saved_models/Engraulis_mordax_future.h5')
par_cal_model_future = load_model('deploy_webapp/saved_models/Paralichthys_californicus_future.h5')
sco_jap_model_future = load_model('deploy_webapp/saved_models/Paralichthys_californicus_future.h5')
thu_ala_model_future = load_model('deploy_webapp/saved_models/Thunnus_alalunga_future.h5')
xip_gla_model_future = load_model('deploy_webapp/saved_models/Xiphias_gladius_future.h5')


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

@app.route("/cit_sor_pred", methods = ['POST'])
def predict_csor():
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
        inRas=gdal.Open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif')
        myarray=inRas.ReadAsArray()

        len_pd=np.arange(len(df))
        lon=df["deci_lon"]
        lat=df["deci_lat"]
        lon=lon.values
        lat=lat.values

        row=[]
        col=[]

        src=rasterio.open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif', crs= 'espg: 4326')
        
        for i in len_pd:
            row_n, col_n = src.index(lon[i], lat[i])# spatial --> image coordinates
            row.append(row_n)
            col.append(col_n)
        
        mean_std=pd.read_csv('data/modified_data/stacked_bio_oracle/env_bio_mean_std.txt',sep="\t")
        mean_std=mean_std.to_numpy()
        
        X=[]
        for j in range(0,9):
            print(j)
            band=myarray[j]
            x=[]
            
            for i in range(0,len(row)):
                value= band[row[i],col[i]]
                if value <-1000:
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/csor_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/csor_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/csor_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = cit_sor_model.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            csor_metric  = eval_pres.iloc[0]
            csor_metric = csor_metric.to_frame()
            result = csor_metric.append(resultdf)
            result = result.to_string()
        
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

@app.route("/eng_mor_pred")
def eng_mor_pred():
    return render_template("eng_mor_pred.html")

@app.route("/eng_mor_pred", methods = ['POST'])
def predict_emor():
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
        inRas=gdal.Open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif')
        myarray=inRas.ReadAsArray()

        len_pd=np.arange(len(df))
        lon=df["deci_lon"]
        lat=df["deci_lat"]
        lon=lon.values
        lat=lat.values

        row=[]
        col=[]

        src=rasterio.open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif', crs= 'espg: 4326')
        
        for i in len_pd:
            row_n, col_n = src.index(lon[i], lat[i])# spatial --> image coordinates
            row.append(row_n)
            col.append(col_n)
        
        mean_std=pd.read_csv('data/modified_data/stacked_bio_oracle/env_bio_mean_std.txt',sep="\t")
        mean_std=mean_std.to_numpy()
        
        X=[]
        for j in range(0,9):
            print(j)
            band=myarray[j]
            x=[]
            
            for i in range(0,len(row)):
                value= band[row[i],col[i]]
                if value <-1000:
                    value=np.nan
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
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/emor_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/emor_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/emor_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = eng_mor_model.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            emor_metric  = eval_pres.iloc[1]
            emor_metric= emor_metric.to_frame()
            result = emor_metric.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

@app.route("/par_cal_pred")
def par_cal_pred():
    return render_template("par_cal_pred.html")

@app.route("/par_cal_pred", methods = ['POST'])
def predict_pcal():
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
        inRas=gdal.Open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif')
        myarray=inRas.ReadAsArray()

        len_pd=np.arange(len(df))
        lon=df["deci_lon"]
        lat=df["deci_lat"]
        lon=lon.values
        lat=lat.values

        row=[]
        col=[]

        src=rasterio.open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif', crs= 'espg: 4326')
        
        for i in len_pd:
            row_n, col_n = src.index(lon[i], lat[i])# spatial --> image coordinates
            row.append(row_n)
            col.append(col_n)
        
        mean_std=pd.read_csv('data/modified_data/stacked_bio_oracle/env_bio_mean_std.txt',sep="\t")
        mean_std=mean_std.to_numpy()
        
        X=[]
        for j in range(0,9):
            print(j)
            band=myarray[j]
            x=[]
            
            for i in range(0,len(row)):
                value= band[row[i],col[i]]
                if value <-1000:
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/pcal_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/pcal_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/pcal_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = par_cal_model.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            pcal_metric  = eval_pres.iloc[2]
            pcal_metric= pcal_metric.to_series()
            result = pcal_metric.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

@app.route("/sco_jap_pred")
def sco_jap_pred():
    return render_template("sco_jap_pred.html")

@app.route("/sco_jap_pred", methods = ['POST'])
def predict_sjap():
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
        inRas=gdal.Open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif')
        myarray=inRas.ReadAsArray()

        len_pd=np.arange(len(df))
        lon=df["deci_lon"]
        lat=df["deci_lat"]
        lon=lon.values
        lat=lat.values

        row=[]
        col=[]

        src=rasterio.open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif', crs= 'espg: 4326')
        
        for i in len_pd:
            row_n, col_n = src.index(lon[i], lat[i])# spatial --> image coordinates
            row.append(row_n)
            col.append(col_n)
        
        mean_std=pd.read_csv('data/modified_data/stacked_bio_oracle/env_bio_mean_std.txt',sep="\t")
        mean_std=mean_std.to_numpy()
        
        X=[]
        for j in range(0,9):
            print(j)
            band=myarray[j]
            x=[]
            
            for i in range(0,len(row)):
                value= band[row[i],col[i]]
                if value <-1000:
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/sjap_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/sjap_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/sjap_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = sco_jap_model.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            sjap_metric= sjap_metric.to_frame()
            result = sjap_metric.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"
    
@app.route("/thu_ala_pred")
def thu_ala_pred():
    return render_template("thu_ala_pred.html")

@app.route("/thu_ala_pred", methods = ['POST'])
def predict_tala():
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
        inRas=gdal.Open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif')
        myarray=inRas.ReadAsArray()

        len_pd=np.arange(len(df))
        lon=df["deci_lon"]
        lat=df["deci_lat"]
        lon=lon.values
        lat=lat.values

        row=[]
        col=[]

        src=rasterio.open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif', crs= 'espg: 4326')
        
        for i in len_pd:
            row_n, col_n = src.index(lon[i], lat[i])# spatial --> image coordinates
            row.append(row_n)
            col.append(col_n)
        
        mean_std=pd.read_csv('data/modified_data/stacked_bio_oracle/env_bio_mean_std.txt',sep="\t")
        mean_std=mean_std.to_numpy()
        
        X=[]
        for j in range(0,9):
            print(j)
            band=myarray[j]
            x=[]
            
            for i in range(0,len(row)):
                value= band[row[i],col[i]]
                if value <-1000:
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/tala_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/tala_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/tala_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = thu_ala_model.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, columns=['result'])
            tala_metric  = eval_pres.iloc[4]
            tala_metric= tala_metric.to_frame()
            result = tala_metric.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"
    
@app.route("/xip_gla_pred")
def xip_gla_pred():
    return render_template("xip_gla_pred.html")

@app.route("/xip_gla_pred", methods = ['POST'])
def predict_xgla(): 
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
        inRas=gdal.Open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif')
        myarray=inRas.ReadAsArray()

        len_pd=np.arange(len(df))
        lon=df["deci_lon"]
        lat=df["deci_lat"]
        lon=lon.values
        lat=lat.values

        row=[]
        col=[]

        src=rasterio.open('data/modified_data/stacked_bio_oracle/bio_oracle_stacked.tif', crs= 'espg: 4326')
        
        for i in len_pd:
            row_n, col_n = src.index(lon[i], lat[i])# spatial --> image coordinates
            row.append(row_n)
            col.append(col_n)
        
        mean_std=pd.read_csv('data/modified_data/stacked_bio_oracle/env_bio_mean_std.txt',sep="\t")
        mean_std=mean_std.to_numpy()
        
        X=[]
        for j in range(0,9):
            print(j)
            band=myarray[j]
            x=[]
            
            for i in range(0,len(row)):
                value= band[row[i],col[i]]
                if value <-1000:
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/xgla_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/xgla_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/xgla_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = xip_gla_model.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            xgla_metric  = eval_pres.iloc[5]
            xgla_metric= xgla_metric.to_frame()
            result = xgla_metric.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

#Predictions: Future
@app.route("/cit_sor_fut_pred")
def cit_sor_fut_pred():
    return render_template("cit_sor_fut_pred.html")

@app.route("/cit_sor_fut_pred", methods = ['POST'])
def predict_csorf():
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
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
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/csor_future_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/csor__future_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/csor_future_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = cit_sor_model_future.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            csor_metric_future  = eval_fut.iloc[0]
            csor_metric_future = csor_metric_future.to_frame()
            result = csor_metric_future.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"  
    else:
        return "Invalid Input"

@app.route("/eng_mor_fut_pred")
def eng_mor_fut_pred():
    return render_template("eng_mor_fut_pred.html")

@app.route("/eng_mor_fut_pred", methods = ['POST'])
def predict_emorf():
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
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
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/emor_future_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/emor__future_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/emor_future_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = eng_mor_model_future.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            emor_metric_future  = eval_fut.iloc[1]
            emor_metric_future = emor_metric_future.to_frame()
            result = emor_metric_future.append(resultdf)
            result  = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

@app.route("/par_cal_fut_pred")
def par_cal__fut_pred():
    return render_template("par_cal_fut_pred.html")

@app.route("/par_cal_fut_pred", methods = ['POST'])
def predict_pcalf():
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]


    if not df.empty:
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
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/pcal_future_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/pcal__future_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/pcal_future_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = par_cal_model_future.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            pcal_metric_future  = eval_fut.iloc[2]
            pcal_metric_future = pcal_metric_future.to_frame()
            result  = pcal_metric_future.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

@app.route("/sco_jap_fut_pred")
def sco_jap_fut_pred():
    return render_template("sco_jap_fut_pred.html")

@app.route("/sco_jap_fut_pred", methods = ['POST'])
def predict_sjapf():
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
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
                    value=np.nan
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
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/sjap_future_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/sjap__future_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/sjap_future_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = sco_jap_model_future.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            sjap_metric_future  = eval_fut.iloc[3]
            sjap_metric_future = sjap_metric_future.to_frame()
            result = sjap_metric_future.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

@app.route("/thu_ala_fut_pred")
def thu_ala_fut_pred():
    return render_template("thu_ala_fut_pred.html")

@app.route("/thu_ala_fut_pred", methods = ['POST'])
def predict_talaf():
    #taking in user input, making a dataframe
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
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
                    value=np.nan
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
        
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/tala_future_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/tala__future_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/tala_future_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = thu_ala_model_future.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])
            tala_metric_future  = eval_fut.iloc[4]
            tala_metric_future = tala_metric_future.to_frame()
            result = tala_metric_future.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

@app.route("/xip_gla_fut_pred")
def xip_gla_fut_pred():
    return render_template("xip_gla_fut_pred.html")

@app.route("/xip_gla_fut_pred", methods = ['POST'])
def predict_xglaf():
    #taking in user input, making a dataframe
    lat = request.form.get('latitudechange')
    latitude = float(lat)
    lon = request.form.get('longitudechange')
    longitude = float(lon)
    items = {"deci_lat": [latitude], "deci_lon": [longitude]}
    df = pd.DataFrame(items)

    df = df[ (df['deci_lat']< 90.) & (df['deci_lat'] > -90.)]
    df = df[ (df['deci_lon']< 180.) & (df['deci_lon'] > -180.) ]

    if not df.empty:
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
                    value=np.nan
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
        df=df.dropna(axis=0, how='any')

        if not df.empty:
            input_X=df.loc[:,0:8]
            row=df[9]
            col=df[10]
            
            row_col=pd.DataFrame({"row":row,"col":col})
            
            input_X=input_X.values
            
            row=row.values
            col=col.values
            
            prediction_array=np.save('deploy_webapp/predictions/xgla_future_prediction_array.npy',input_X)
            prediction_pandas=row_col.to_csv('deploy_webapp/predictions/xgla__future_prediction_row_col.csv')
            
            input_X=np.load('deploy_webapp/predictions/xgla_future_prediction_array.npy')
            df=pd.DataFrame(input_X)
            
            new_band=myarray[1].copy()
            new_band.shape
            
            new_values = xip_gla_model_future.predict(x=input_X,verbose=0) ###predict output value
            new_band_values=[]
            
            for i in new_values:
                new_value=i[1]
                new_band_values.append(new_value)
            new_band_values=np.array(new_band_values)
            
            my_string = "Predicted Probability: "
            resultdf = pd.DataFrame(new_band_values, index=['predicted_result'])    
            xgla_metric_future  = eval_fut.iloc[5]
            xgla_metric_future = xgla_metric_future.to_frame()
            result = xgla_metric_future.append(resultdf)
            result = result.to_string()
            return result
        else: 
            return "Land Coordinate!"
    else:
        return "Invalid Entry!"

if __name__ == "__main__":
    app.run(debug=True)
    
