import numpy as np
from tensorflow.keras.models import load_model
import joblib

from flask import Flask, render_template, session, url_for, redirect
from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"

class FlowerForm(FlaskForm):
    sepal_length = TextField("Sepal Length")
    sepal_width = TextField("Sepal Width")
    petal_length = TextField("Petal Length")
    petal_width = TextField("Petal Width")

    submit = SubmitField("Analyze")

def return_prediction(model, scaler, sample_json):

    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]

    flower = [[s_len, s_wid, p_len, p_wid]]
    classes = ['setosa', 'versicolor', 'virginica']

    flower = scaler.transform(flower)

    class_ind = np.argmax(model.predict(flower), axis=-1)[0]

    return classes[class_ind]

@app.route("/", methods=["GET", "POST"])
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['sepal_length'] = form.sepal_length.data
        session['sepal_width'] = form.sepal_width.data
        session['petal_length'] = form.petal_length.data
        session['petal_width'] = form.petal_width.data

        return redirect(url_for("prediction"))
    return render_template("home.html", form=form)

flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

@app.route("/prediction")
def prediction():
    content = {}
    content["sepal_length"] = float(session['sepal_length'])
    content["sepal_width"] = float(session['sepal_width'])
    content["petal_length"] = float(session['petal_length'])
    content["petal_width"] = float(session['petal_width'])

    results = return_prediction(flower_model, flower_scaler, content)

    return render_template("prediction.html", results=results)

if __name__ == "__main__":
    app.run()
