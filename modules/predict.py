import os
import glob
import dill
import json
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
file_list = os.listdir(f'{path}/data/models')


def predict():
    with open(f'{path}/data/models/{file_list[-1]}', 'rb') as file:
        model = dill.load(file)

    df_predictions = pd.DataFrame(columns=['car_id', 'price_category'])

    for sample_file in glob.glob(f'{path}/data/test/*.json'):
        with open(sample_file) as sample_object:
            form = json.load(sample_object)
        df_pred = pd.DataFrame.from_dict([form])
        y = model.predict(df_pred)
        x = {'car_id': form['id'], 'price_category': y}
        df_pred_new = pd.DataFrame([x])
        df_predictions = df_predictions._append(df_pred_new, ignore_index=True)

    pred_filename = f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv'

    df_predictions.to_csv(pred_filename)


predict()
