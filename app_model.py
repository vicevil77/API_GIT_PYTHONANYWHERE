from flask import Flask, jsonify, request,render_template
# render template es para renderizar el html
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import subprocess

#EJEMPLO PARA LA PRUEBA:
#ejemplo en clase :  /api/v1/predict?work_year=5&experience_level=Senior&employment_type=Full-time&job_title=Software%20Engineer&salary_currency=USD&salary_in_usd=50000&employee_residence=USA&remote_ratio=0.5&company_location=USA&company_size=100-500
# REENTRENO: /api/v1/retrain/

root_path= "http://127.0.0.1:5000/"

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    #return  "<h1>Bienvenido al modelo predictorio de salarios de Data Scientit: /api/v1/predit o /api/v1/retrain </h1>"
    return render_template('home.html')

with open(r"D:\Cursos\REPOSITORIOS\DESPLIEGUE_API_SALARIOS_copia\data\salary_pipeline.pkl", 'rb') as file:
    pipeline = pickle.load(file)

@app.route('/api/v1/predict', methods=['GET', 'POST'])
def predict():
    
    # obtener los datos de la solicitud GET 
    data = {
        "work_year": request.args.get('work_year'),
        "experience_level": request.args.get('experience_level'),
        "employment_type": request.args.get('employment_type'),
        "job_title": request.args.get('job_title'),
        "salary_currency": request.args.get('salary_currency'),
        "salary_in_usd": request.args.get('salary_in_usd'),
        "employee_residence": request.args.get('employee_residence'),
        "remote_ratio": request.args.get('remote_ratio'),
        "company_location": request.args.get('company_location'),
        "company_size": request.args.get('company_size')
        }
           
    # establecer valores predeterminados para los datos faltantes
    default_values = {
        "work_year": 2021,
        "experience_level": 'MI',
        "employment_type": 'FT',
        "job_title": 'Data Scientist',
        "salary_currency": 'USD',
        "salary_in_usd": 79833,
        "employee_residence": 'US',
        "remote_ratio": 0,
        "company_location": 'US',
        "company_size": 'S'
    }

    for key in data:
        if data[key] is None:
            data[key] = default_values[key]
        
         
    # convertir los datos a DataFrame
    data_df = pd.DataFrame(data, index=[0])
    # preprocesar los datos
    preprocessed_data = pipeline.named_steps['preprocessor'].transform(data_df)
    prediction = pipeline.named_steps['model'].predict(preprocessed_data)
    
    #preprocessed_data = pipeline['preprocessor'].transform(data_df)
    # hacer la predicción
    #prediction = pipeline['model'].predict(preprocessed_data)
    # renderizar la plantilla HTML
    return render_template('predict.html', prediction=prediction[0])

@app.route('/api/v1/retrain', methods=['GET', "POST"])
def retrain():
    # leer los datos del archivo CSV
    data = pd.read_csv(r'D:\Cursos\REPOSITORIOS\DESPLIEGUE_API_SALARIOS_copia\data\ds_salaries.csv')
    # dividir los datos en características y variable objetivo
    X = data.drop('salary', axis=1)
    y = data['salary']
    # preprocesar las características del pipieline
    X_preprocessed = pipeline['preprocessor'].transform(X)
    # dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2)
    # reentrenar el modelo
    pipeline['model'].fit(X_train, y_train)
    # calcular el error cuadrático medio y el error porcentual absoluto medio
    rmse = np.sqrt(mean_squared_error(y_test, pipeline['model'].predict(X_test)))
    mape = mean_absolute_percentage_error(y_test, pipeline['model'].predict(X_test))  
     # renderizar la plantilla HTML
    return render_template('retrain.html', rmse=rmse, mape=mape)
    # guardar el modelo reentrenado
    #with open(r'D:\Cursos\REPOSITORIOS\DESPLIEGUE_API_SALARIOS_copia\data\RETRAIN', 'wb') as file:
            #pickle.dump(pipeline['model'], file)
    #return {'message': 'Model retrained successfully', 'RMSE': rmse, 'MAPE': mape}
   
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