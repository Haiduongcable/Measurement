import os
import threading
import requests
import time
from tqdm import tqdm

class MyThread(threading.Thread):
    def run(self):
        os.system('dockerd')

os.system(
    '''echo '{ "storage-driver": "btrfs" }' | tee /etc/docker/daemon.json''')
os.system('cat /etc/docker/daemon.json')


thread = MyThread()
thread.daemon = True
thread.start()
# os.environ["DOCKER_CLIENT_TIMEOUT"] = '5000'
# os.environ["COMPOSE_HTTP_TIMEOUT"] = '5000'
# # os.environ["COMPOSE_MOUNT_DIR"] = '/home/haiduong/Documents/VIN_BRAIN/Measurement/Log_remain_tb_testset'
# os.environ["COMPOSE_MOUNT_DIR"] = '/home'
# os.system('docker login -u measurement-team -p gnqmriW5UEGqknEkxPZ7TubyVQd+4mxF vinbrain.azurecr.io')
# # os.system('wget https://raw.githubusercontent.com/Haiduongcable/Measurement/main/docker-compose.yaml -O docker-compose.yaml')
# print('compose up run')
# os.system('docker-compose up -d')
# print('compose up run finish')





def predict(image_path):
   
    #image_path = "https://vbagenda.blob.core.windows.net/xray/64542.jpg"
    # image_path = "/home/0a1fb6670236406b0bcf160eec59163edc14072fa79fe6047b79477115dcefdc.jpg"

    headers = {
        'Content-Type': 'application/json',
        # 'x-auth-token': '1234566'
    }
    url = "http://127.0.0.1:8080/images"
    data = {
        "imageUrl": image_path,
        "mode": "detection",
        "additional": {
            "histogram_adjustment": True
        }
    }
    response = None
    try:
        response = requests.post(url, json=data, headers=headers)
    except Exception as e:
        print(e)
    return response




path_folder = "/home/haiduong/Documents/VIN_BRAIN/Measurement/Log_remain_tb_testset"
l_image = os.listdir(path_folder)

for image in tqdm(l_image[:200]):
    path_image = '/home/' + image
    print('_'*50 + 'predict' + '_'*50)
    response = predict(path_image)
    try:
        print('response', response.json())
    except:
        print('response', response)

# print('_'*50 + 'predict' + '_'*50)
# response = predict('')
# try:
#     print('response', response.json())
# except:
#     print('response', response)

# print('_'*50 + 'predict' + '_'*50)
# response = predict('')
# try:
#     print('response', response.json())
# except:
#     print('response', response)


# print('_'*50 + 'docker-compose down' + '_'*50)

