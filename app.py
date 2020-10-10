import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import pickle
import matplotlib.pyplot as plt
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

app = Flask(__name__)
# model, forecast = pickle.load(open('model.pkl', 'rb'))

with open('model.pickle', 'rb') as f:
    supplies_test, supplies_forecast, model = pickle.load(f)

supplies = pd.read_csv('Supplies3.csv', parse_dates=[0], index_col=[0])

@app.route('/')
def home():
    output = supplies_forecast
    # output = model.forecast(steps = 35)[0]
    return render_template('index.html', prediction_text='{}'.format(output))

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Order amount should be $ ()'.format(output))

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    # fig = plt.plot(supplies_forecast)
    axis = fig.add_subplot(1, 1, 1)
    # xs = range(35)
    # ys = [supplies_forecast for x in xs]
    axis.plot(supplies_forecast)
    return fig

if __name__ == "__main__":
    app.run(debug=True)