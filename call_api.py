import os
import numpy as np
import argparse
import json
import pandas as pd
import time
import shutil
import zipfile
from azureml.core import Run
import requests
import threading
import psutil



def refine_image_path(image_name):
    global image_dir
    for prefix in ['xray-testing-data', 'Phoi_Labelling_Data', 'vinmec/anonymized-images']:
        path = os.path.join(image_dir, prefix, image_name)
        if os.path.exists(path):
            return path
    return image_name


class MyThread(threading.Thread):
    def run(self):
        os.system('/bin/bash -c "source /opt/intel/openvino_2021/bin/setupvars.sh && cd /code/build/chest-detection && ./chest-detection"')


    


def convert_json_predict(ret_obs, threshold = 0.75, type_disease = 'TUB'):
    ret_obs = ret_obs['observations']
    tub_obs = list(filter(lambda x:x["code"] == type_disease, ret_obs))[0]
    confidence_score = tub_obs["confidentScore"]
    threshold = tub_obs["threshold"]
    print("Value threshold", threshold)
    if confidence_score >= threshold:
        predict = 1
    else:
        predict = 0
    return predict


def predict(image_path):

    headers = {
        'Content-Type': 'application/json',
        'x-auth-token': '1234566'
    }
    url = "http://127.0.0.1:8080/images"
    data = {"imageUrl": image_path}
    try:
        response = requests.post(url, json=data, headers=headers)
    except Exception as e:
        print(e)
    return response


def call_api(image_path):


    header = {'x-auth-token': '123456', 'Content-Type': 'application/json'}
    try:
        raw = requests.post("http://127.0.0.1:8080/images",
                            data=json.dumps({'imageUrl': image_path}), headers=header)
        if raw.status_code == 200:
            # print('ok')
            if json.loads(raw.text) is None:
                return raw.json()
            return json.loads(raw.text)
    except:
        raise ValueError(image_path)


def call_api(image_path):
    # image_path = "https://api-int.draid.ai/dicom/studies/2.25.160091743131929387903451525446410949934/series/2.25.148244682741390089451659848730132728740/instances/2.25.97039210792747508295850276140512446062/frames/1/rendered?tenantCode=VB&accept=image/png"
    headers = {
        'Content-Type': 'application/json',
        # 'x-auth-token': '1234566'
    }
    url = "http://127.0.0.1:8080/images"
    data = {
        "imageUrl": image_path,
        "mode": "detection",
        "additional": {
            "histogram_adjustment": False
        }
    }
    try:

        raw = requests.post("http://127.0.0.1:8080/images",
                            json=data, headers=headers)
        print(raw)
        
        if raw.status_code == 200:
            # print('ok')
            if json.loads(raw.text) is None:
                return raw.json()
            return json.loads(raw.text)
        
    except Exception as e:
        # raise ValueError(image_path)
        print('image_path', image_path)
        print('Exception', e)


total_image = pd.read_csv("/home/haiduong/Documents/VIN_BRAIN/Measurement/Tuberculosis/Convert_data/testing_tuberculosis.csv")
l_total_image = []
for index, row in total_image.iterrows():
    name_image = row["Images"].split("/")[-1]
    l_total_image.append(name_image)

done_image = pd.read_csv("/home/haiduong/Documents/VIN_BRAIN/Measurement/Tuberculosis/Convert_data/Done/Done_09_07.csv")
l_done_image = []
for index, row in done_image.iterrows():
    name_image = row["Images"].split("/")[-1]
    l_done_image.append(name_image)


y_pred = []
y_true = []
processed_images = []
thresholds = []
print_step = 50
processed = 0

nbatch = 0
for index, row in total_image.iterrows():
    name_image = row["Images"].split("/")[-1]
    if name_image in l_done_image:
        continue
    image_path = "/home/" + name_image
    try:
        ret_obs = call_api(image_path)
        print(ret_obs)
        predict = convert_json_predict(ret_obs, threshold = 0.75, type_disease = 'TUB')
    except Exception as e:
        if(os.path.exists(image_path)):
            print(
                "Failed at {} But it's already in disk =>> API-ER".format(name_image))
        else:
            print("FAILED at {} and it's not in disk".format(name_image))
        print('error:', e)
        continue
    processed += 1
    nbatch += 1

    y_pred.append(predict)
    y_true.append(int(row['TB']))

    e_time = time.time()
    processed_images.append(name_image)

# avg_pred_time = avg_pred_time/nbatch     
# overall_used_memory = psutil.virtual_memory()[2] - memory_footprint
memory_footprint = 0
overall_used_memory = 0
avg_pred_time = 0
ret_df = pd.DataFrame()
ret_df["Images"] = processed_images
ret_df["Preds"] = np.asarray(y_pred, dtype=np.float32)
ret_df["Truths"] = np.asarray(y_true, dtype=np.float32)

ret_df["Average_Time"] = [avg_pred_time,]*len(y_pred)
ret_df["Memory init"] = [memory_footprint,]*len(y_pred)
ret_df["Overall used memory"] = [overall_used_memory,]*len(y_pred)
ret_df.to_csv("Done.csv")
