from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle
from src.utils import load_model
from sklearn.pipeline import Pipeline

PREDICTION_FOLDER = "batch_prediction"
PREDICTION_CSV = "prediction_csv"
PREDICTION_FILE = "output.csv"
FEATURE_ENG_FOLDER = "feature_eng"

ROOT_DIR = os.getcwd()
BATCH_PREDICTION = os.path.join(ROOT_DIR, PREDICTION_FOLDER, PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR, PREDICTION_FOLDER, FEATURE_ENG_FOLDER)

class batch_prediction:
    def __init__(self, input_file_path,
                 model_file_path, transformer_file_path,
                 feature_engineering_file_path) -> None:
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

    def start_batch_prediction(self):
        try:
            # Load the feature engineering pipeline path
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            # Load the data transformation pipeline file path

            with open(self.transformer_file_path, 'rb') as f:
                processor = pickle.load(f)

            # Load the model separatly

            model = load_model(self.model_file_path)

            # create a feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ("feature_engineering", feature_pipeline)

            ])

            df = pd.read_csv(self.input_file_path)

            df.to_csv("df_zomato_delivery_time_prediction.csv")

            # Apply feature engineering pipeline steps

            df = feature_engineering_pipeline.transform(df)

            df.to_csv("feature_engineering.csv")

            FEATURE_ENGINEERING_PATH = FEATURE_ENG
            os.makedirs(FEATURE_ENGINEERING_PATH, exist_ok=True)

            file_path = os.path.join(FEATURE_ENGINEERING_PATH, 'batch_feature_eng.csv')

            df.to_csv(file_path, index = False)

            # time_taken column dropped
            df = df.drop('Time_taken (min)', axis = 1)
            df.to_csv("time_taken_dropped.csv")

            transform_data = processor.transform(df)

            file_path = os.path.join(FEATURE_ENGINEERING_PATH, 'processor.csv')

            predictions = model.predict(transform_data)

            df_prediction = pd.DataFrame(predictions, columns=['prediction'])


            BATCH_PREDICTION_PATH = BATCH_PREDICTION
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH, PREDICTION_FILE)

            df_prediction.to_csv(csv_path, index = False)
            logging.info(f"Batch prediuction done")


        except Exception as e:
            raise CustomException(e, sys)