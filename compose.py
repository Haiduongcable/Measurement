import os
import threading
import requests
import time


class MyThread(threading.Thread):
    def run(self):
        os.system('dockerd')

os.system(
    '''echo '{ "storage-driver": "btrfs" }' | tee /etc/docker/daemon.json''')
os.system('cat /etc/docker/daemon.json')


thread = MyThread()
thread.daemon = True
thread.start()
os.environ["DOCKER_CLIENT_TIMEOUT"] = '5000'
os.environ["COMPOSE_HTTP_TIMEOUT"] = '5000'
os.environ["COMPOSE_MOUNT_DIR"] = '/home'
os.system('docker login -u measurement-team -p gnqmriW5UEGqknEkxPZ7TubyVQd+4mxF vinbrain.azurecr.io')
# os.system('wget https://raw.githubusercontent.com/Haiduongcable/Measurement/main/docker-compose.yaml -O docker-compose.yaml')
print('compose up run')
os.system('docker-compose up -d')
print('compose up run finish')
time.sleep(400)


def predict(image_path):
    image_path = "/home/haiduong/Documents/VIN_BRAIN/Measurement/111b9edb2ed66ac22ae469d64c1e65e8d7ba733fe6c37638b928feec9d01fa2e.png"

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


print('_'*50 + 'predict' + '_'*50)
response = predict('')
try:
    print('response', response.json())
except:
    print('response', response)


print('_'*50 + 'predict' + '_'*50)
response = predict('')
try:
    print('response', response.json())
except:
    print('response', response)

print('_'*50 + 'predict' + '_'*50)
response = predict('')
try:
    print('response', response.json())
except:
    print('response', response)


print('_'*50 + 'docker-compose down' + '_'*50)
