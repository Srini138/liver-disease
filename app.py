# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:47:02 2021

@author: SHANMATHU
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open("liver_disease_pred.pkl" ,'rb'))
sc=pickle.load(open("std_scl.pkl",'rb'))

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')


@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
        age=request.form['age']
        Gender=request.form["gender"]
        Total_Bilirubin=request.form["Total_Bilirubin"]
        Alkaline_Phosphotase=request.form["Alkaline_Phosphotase"]
        Alamine_Aminotransferase=request.form["Alamine_Aminotransferase"]
        Aspartate_Aminotransferase=request.form["Aspartate_Aminotransferase"]
        Total_Protiens=request.form["Total_Protiens"]
        Albumin=request.form["Albumin"]
        Albumin_and_Globulin_Ratio=request.form["Albumin_and_Globulin_Ratio"]
        
        if Gender=='Male':
            Gender=1
        else :
            Gender=0
        a=sc.transform(np.array([age,Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase, Aspartate_Aminotransferase,Total_Protiens, Albumin,Albumin_and_Globulin_Ratio]).reshape(1,-1))
        b=[[Gender]]
        out=np.hstack([a,b])
       # inp=[[age,Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase, Aspartate_Aminotransferase,Total_Protiens, Albumin,Albumin_and_Globulin_Ratio,Gender]]
        pred=model.predict(out)
        outp=pred[0]
        return render_template('predict.html', pred = outp)



if __name__ == '__main__':
	app.run(debug = True)