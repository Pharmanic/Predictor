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
    supplies_test, supplies_forecast, supplies_model_fit = pickle.load(f)

# supplies = pd.read_csv('Supplies.csv', parse_dates=[0], index_col=[0])

@app.route('/')
def home():
    # output = supplies.values
    output = supplies_forecast
    # output = sup_forecast
    # output = model.forecast(steps = 35)[0]
    return render_template('index.html', prediction_text='{}'.format(output))

@app.route('/plot_png', methods=['POST', 'GET'])
def plot_png():
    if request.method == 'POST':
        steps = request.form['steps']

    fig = create_figure(int(steps))
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(steps):
    fig = Figure()
    # fig = plt.plot(supplies_forecast)
    axis = fig.add_subplot(1, 1, 1)
    model_fit = supplies_model_fit.fit()
    sup_forecast = model_fit.forecast(steps = steps)[0]
    # axis.plot(supplies_forecast)
    axis.plot(sup_forecast)
    return fig



if __name__ == "__main__":
    app.run(debug=True)