from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import subprocess
import logging

root_path= "/home/JBLapiweb/bike_predictor/"

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return  "<h1>Hello! Here is you solution to predict public bike facility use in your city</h1>"

@app.route('/api/v1/predict', methods= ['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET

    # Configure logging
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    # Usage in your application
    try:
        model = pickle.load(open(root_path +'bike_model.pkl','rb'))
        pass
    except Exception as e:
        logging.error("An error occurred", exc_info=True)

    holiday = request.args.get('holiday', 0)
    workingday = request.args.get('workingday', 1)
    weather = request.args.get('weather', 0)
    temp = request.args.get('temp', 0.5)
    humidity = request.args.get('humidity', 0.65)
    hour = request.args.get('hour', None)
    month = request.args.get('month', None)
    weekday = request.args.get('weekday', None)
    windspeed_log = request.args.get('windspeed', 0)        ### PREGUNTAR CÓMO HACER MODIFICACIONES A FEATURES

    features = [holiday, workingday, weather, temp, humidity, hour, month, weekday, windspeed_log]

    # print(tv,radio,newspaper)   # Buscar cómo hacer el print de las features !!!!

    if hour is None or month is None or weekday is None:
        return "Args empty, the data are not enough to predict. \n Please specify at least hour, month and weekday"
    else:
        prediction = model.predict([[float(feature) for feature in features]])     
    return jsonify({'predictions': prediction[0]})

@app.route('/api/v1/retrain/', methods=['GET'])
def retrain(): # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists(root_path +"data/bike_train.csv"):
        data = pd.read_csv(root_path +'data/bike_train.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['count']),
                                                        data['count'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['count']), data['count'])
        pickle.dump(model, open(root_path +'bike_model.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

@app.route('/webhook_2024', methods=['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = root_path
    servidor_web = '/var/www/jblapiweb_pythonanywhere_com_wsgi.py' 

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': "This directory doesn't exist"}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull'], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f"A Git pull was made in the Repository: {repo_name}"}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error while Git pull in: {repo_name}'}), 500
        else:
            return jsonify({'message': 'No information found about the repository in the load (payload)'}), 400
    else:
        return jsonify({'message': 'No JSON found in the request'}), 400
    
    
if __name__ == '__main__':
    app.run()