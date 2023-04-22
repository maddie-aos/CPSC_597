from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/present")
def present():
    return render_template("present.html")

@app.route("/future")
def future():
    return render_template("future.html")


#Present distributions
@app.route("/cit_sor_pres")
def cit_sor_pres():
    return render_template("cit_sor_pres.html")

@app.route("/cit_sor_dist")
def cit_sor_dist():
    return render_template("cit_sor_dist.html")

@app.route("/cit_sor_pred")
def cit_sor_pred():
    return render_template("cit_sor_pred.html")



@app.route("/eng_mor_pres")
def eng_mor_pres():
    return render_template("eng_mor_pres.html")

@app.route("/eng_mor_dist")
def eng_mor_dist():
    return render_template("eng_mor_dist.html")

@app.route("/eng_mor_pred")
def eng_mor_pred():
    return render_template("eng_mor_pred.html")



@app.route("/par_cal_pres")
def par_cal_pres():
    return render_template("par_cal_pres.html")

@app.route("/par_cal_dist")
def par_cal_dist():
    return render_template("par_cal_dist.html")

@app.route("/par_cal_pred")
def par_cal_pred():
    return render_template("par_cal_pred.html")


@app.route("/sco_jap_pres")
def sco_jap_pres():
    return render_template("sco_jap_pres.html")

@app.route("/sco_jap_dist")
def sco_jap_dist():
    return render_template("sco_jap_dist.html")

@app.route("/sco_jap_pred")
def sco_jap_pred():
    return render_template("sco_jap_pred.html")




@app.route("/thu_ala_pres")
def thu_ala_pres():
    return render_template("thu_ala_pres.html")

@app.route("/thu_ala_dist")
def thu_ala_dist():
    return render_template("thu_ala_dist.html")

@app.route("/thu_ala_pred")
def thu_ala_pred():
    return render_template("thu_ala_pred.html")



@app.route("/xip_gla_pres")
def xip_gla_pres():
    return render_template("xip_gla_pres.html")

@app.route("/xip_gla_dist")
def xip_gla_dist():
    return render_template("xip_gla_dist.html")

@app.route("/xip_gla_pred")
def xip_gla_pred():
    return render_template("xip_gla_pred.html")



#Future Distributions
@app.route("/cit_sor_fut")
def cit_sor_fut():
    return render_template("cit_sor_fut.html")

@app.route("/cit_sor_fut_dist")
def cit_sor_fut__dist():
    return render_template("cit_sor_fut_dist.html")

@app.route("/cit_sor_fut_pred")
def cit_sor_fut_pred():
    return render_template("cit_sor_fut_pred.html")



@app.route("/eng_mor_fut")
def eng_mor_fut():
    return render_template("eng_mor_fut.html")

@app.route("/eng_mor_fut_dist")
def eng_mor_fut_dist():
    return render_template("eng_mor_fut_dist.html")

@app.route("/eng_mor_fut_pred")
def eng_mor_fut_pred():
    return render_template("eng_mor_fut_pred.html")



@app.route("/par_cal_fut")
def par_cal_fut():
    return render_template("par_cal_fut.html")

@app.route("/par_cal_fut_dist")
def par_cal_fut_dist():
    return render_template("par_cal_fut_dist.html")

@app.route("/par_cal_fut_pred")
def par_cal__fut_pred():
    return render_template("par_cal_fut_pred.html")



@app.route("/sco_jap_fut")
def sco_jap_fut():
    return render_template("sco_jap_fut.html")

@app.route("/sco_jap_fut_dist")
def sco_jap_fut_dist():
    return render_template("sco_jap_fut_dist.html")

@app.route("/sco_jap_fut_pred")
def sco_jap_fut_pred():
    return render_template("sco_jap_fut_pred.html")




@app.route("/thu_ala_fut")
def thu_ala_fut():
    return render_template("thu_ala_fut.html")

@app.route("/thu_ala_fut_dist")
def thu_ala_fut_dist():
    return render_template("thu_ala_fut_dist.html")

@app.route("/thu_ala_fut_pred")
def thu_ala_fut_pred():
    return render_template("thu_ala_fut_pred.html")



@app.route("/xip_gla_fut")
def xip_gla_fut():
    return render_template("xip_gla_fut.html")

@app.route("/xip_gla_fut_dist")
def xip_gla_fut_dist():
    return render_template("xip_gla_fut_dist.html")

@app.route("/xip_gla_fut_pred")
def xip_gla_fut_pred():
    return render_template("xip_gla_fut_pred.html")




#Loading machine learning models

if __name__ == "__main__":
    app.run(debug=True)