from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("spam_detector.html")


@app.route('/predict',methods=['POST'])
def predict():
    features=[x for x in request.form.values()]
    final=[pd.Series(features)]
    print(features)
    print(final)
    prediction=model.predict(final)
    output= ("your message is: {}".format(b[0]))
    
    return render_template('spam_detector.html',pred='{}'.format(output),bhai="kuch karna h ab")


if __name__ == '__main__':
    app.run(debug=True)