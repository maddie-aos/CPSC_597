from flask import Flask, render_template, request
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model , model_from_json
from keras.optimizers import Adam

app = Flask(__name__)

#loading keras models: present
def load_cit_sor_model():
    json_file = open('results/fish/Citharichthys_sordidus/Citharichthys_sordidus_model.json')
    loaded_model_csor = json_file.read()
    json_file.close()
    load_cit_sor_model = model_from_json(loaded_model_csor)
    load_cit_sor_model.load_weights('results/fish/Citharichthys_sordidus/Citharichthys_sordidus_model.h5')
    load_cit_sor_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

 
def load_eng_mor_model():
    json_file = open('results/fish/Engraulis_mordax/Engraulis_mordax_model.json')
    loaded_model_emor = json_file.read()
    json_file.close()
    load_eng_mor_model = model_from_json(loaded_model_emor)
    load_eng_mor_model.load_weights('results/fish/Engraulis_mordax/Engraulis_mordax_model.h5')
    load_eng_mor_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


def load_par_cal_model():
    json_file = open('results/fish/Paralichthys_californicus/Paralichthys_californicus_model.json')
    loaded_model_pcal = json_file.read()
    json_file.close()
    load_par_cal_model = model_from_json(loaded_model_pcal)
    load_par_cal_model.load_weights('results/fish/Paralichthys_californicus/Paralichthys_californicus_model.h5')
    load_par_cal_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


def load_sco_jap_model():
    json_file = open('results/fish/Scomber_japonicus/Scomber_japonicus_model.json')
    loaded_model_sjap = json_file.read()
    json_file.close()
    load_sco_jap_model = model_from_json(loaded_model_sjap)
    load_sco_jap_model.load_weights('results/fish/Scomber_japonicus/Scomber_japonicus_model.h5')
    load_sco_jap_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_thu_ala_model():
    json_file = open('results/fish/Thunnus_alalunga/Thunnus_alalunga_model.json')
    loaded_model_tala = json_file.read()
    json_file.close()
    load_thu_ala_model = model_from_json(loaded_model_tala)
    load_thu_ala_model.load_weights('results/fish/Thunnus_alalunga/Thunnus_alalunga_model.h5')
    load_thu_ala_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_xip_gla_model():
    json_file = open('results/fish/Xiphias_gladius/Xiphias_gladius_model.json')
    loaded_model_xgla = json_file.read()
    json_file.close()
    load_xip_gla_model = model_from_json(loaded_model_xgla)
    load_xip_gla_model.load_weights('results/fish/Xiphias_gladius/Xiphias_gladius_model.h5')
    load_xip_gla_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])



#loading keras models: future
def load_cit_sor_model_future():
    json_file = open('results/fish_future/Citharichthys_sordidus/Citharichthys_sordidus_future_model.json')
    loaded_model_csorf = json_file.read()
    json_file.close()
    load_cit_sor_model_future = model_from_json(loaded_model_csorf)
    load_cit_sor_model_future.load_weights('results/fish_future/Citharichthys_sordidus/Citharichthys_sordidus_future_model.h5')
    load_cit_sor_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_eng_mor_model_future():
    json_file = open('results/fish_future/Engraulis_mordax/Engraulis_mordax_future_model.json')
    loaded_model_emorf = json_file.read()
    json_file.close()
    load_eng_mor_model_future = model_from_json(loaded_model_emorf)
    load_eng_mor_model_future.load_weights('results/fish_future/Engraulis_mordax/Engraulis_mordax_future_model.h5')
    load_eng_mor_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_par_cal_model_future():
    json_file = open('results/fish_future/Paralichthys_californicus/Paralichthys_californicus_future_model.json')
    loaded_model_pcalf = json_file.read()
    json_file.close()
    load_par_cal_model_future = model_from_json(loaded_model_pcalf)
    load_par_cal_model_future.load_weights('results/fish_future/Paralichthys_californicus/Paralichthys_californicus_future_model.h5')
    load_par_cal_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_sco_jap_model_future():
    json_file = open('results/fish_future/Scomber_japonicus/Scomber_japonicus_future_model.json')
    loaded_model_sjapf = json_file.read()
    json_file.close()
    load_sco_jap_model_future = model_from_json(loaded_model_sjapf)
    load_sco_jap_model_future.load_weights('results/fish_future/Scomber_japonicus/Scomber_japonicus_future_model.h5')
    load_sco_jap_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_thu_ala_model_future():
    json_file = open('results/fish_future/Thunnus_alalunga/Thunnus_alalunga_future_model.json')
    loaded_model_talaf = json_file.read()
    json_file.close()
    load_thu_ala_model_future = model_from_json(loaded_model_talaf)
    load_thu_ala_model_future.load_weights('results/fish_future/Thunnus_alalunga/Thunnus_alalunga_future_model.h5')
    load_thu_ala_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

def load_xip_gla_model_future():
    json_file = open('results/fish_future/Xiphias_gladius/Xiphias_gladius_future_model.json')
    loaded_model_xglaf = json_file.read()
    json_file.close()
    load_xip_gla_model_future = model_from_json(loaded_model_xglaf)
    load_xip_gla_model_future.load_weights('results/fish_future/Xiphias_gladius/Xiphias_gladius_future_model.h5')
    load_xip_gla_model_future.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])





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
@app.route("/cit_sor_fut_pred", methods = ['GET', 'POST'])
def cit_sor_fut_pred():
    return render_template("cit_sor_fut_pred.html")
def predict():
    if request.method == "POST":
        latitude = request.form.get('changelatitude')
        longitude = request.form.get('changelongitude')
        return "Your coordinates are "+latitude + longitude
    #return render_template("cit_sor_fut_pred.html")
    #I NEED HELP HERRREEEEEEEEEEEE ALEX


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
    
