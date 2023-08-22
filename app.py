from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
import os
from werkzeug.utils import secure_filename

from Prediction.batch import batch_prediction
from src.logger import logging
from src.component.data_transformation import DataTransformationConfig
from src.config.configuration import *
from src.pipeline.training_pipeline import Train

feature_enfineering_file_path = FEATURE_ENGG_OBJ_FILE_PATH
transformer_file_path = PREPROCESSING_OBJ_FILE
model_file_path = MODEL_FILE_PATH

UPLOAD_FOLDER = 'batch_Prediction/UPLOADED_CSV_FILE'

app = Flask(__name__, template_folder = 'templates')

ALLOWED_EXTENSION = {'csv'}

# localhost: 5000
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods= ['GET', 'POST'])
def predict_datapoint():
    if(request.method == 'GET'):
        return render_template('form.html')
    
    else:
        data = CustomData(
           Delivery_person_Age= int(request.form.get('Delivery_person_Age')),
           Delivery_person_Ratings= float(request.form.get('Delivery_person_Ratings')),
           Weather_conditions= request.form.get('Weather_conditions'),
           Road_traffic_density= request.form.get('Road_traffic_density'),
           Vehicle_condition= float(request.form.get('Vehicle_condition')),
           multiple_deliveries= float(request.form.get('multiple_deliveries')),
           distance= float(request.form.get('distance')),
           Type_of_order= request.form.get('Type_of_order'),
           Type_of_vehicle= request.form.get('Type_of_vehicle'),
           Festival= request.form.get('Festival'),
           City= request.form.get('City')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)

        result = int(pred[0])

        return render_template('form.html', final_result = result)
    
@app.route('/batch', methods = ['GET', 'POST'])
def perform_batch_prediction():

    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files['csv_file'] # update the key to 'csv_file'
        # Directory path
        directory_path= UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

        # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("csv received and  uploaded")

            # Perform batch prediction using the uploaded file 
            batch = batch_prediction(file_path,
                                     model_file_path,
                                     transformer_file_path,
                                     feature_enfineering_file_path)
            batch.start_batch_prediction()

            output = "Batch Prediction Done"
            return render_template("batch.html", prediction_result=output, prediction_type = 'batch')
        else:
            return render_template("batch.html", prediction_type = 'batch', error = 'Invalid file type')
        
@app.route('/train', methods = ['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()

            return render_template('train.html', message = "Training complete")
        except Exception as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template('index.html', error = error_message)
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port = '8888')
        