from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import subprocess

root_path= "/home/vicevil/taller_despliegue/"

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return  "<h1>Hello, World! This is a simple API for a ML model</h1>"

@app.route('/api/v1/predict', methods= ['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET

    model = pickle.load(open(root_path +'ad_model.pkl','rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])
    
    return jsonify({'predictions': prediction[0]})

@app.route('/api/v1/retrain/', methods=['GET'])
def retrain(): # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists(root_path +"data/Advertising_new.csv"):
        data = pd.read_csv(root_path +'data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        pickle.dump(model, open(root_path +'ad_model.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

@app.route('/webhook_2024', methods=['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = "/home/vicevil/taller_despliegue"
    servidor_web = '/var/www/vicevil_pythonanywhere_com_wsgi.py' 

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
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull'], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400
    
    
if __name__ == '__main__':
    app.run()