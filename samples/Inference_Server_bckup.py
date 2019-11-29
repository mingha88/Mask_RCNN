

import os

import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

from PIL import Image
import threading
import socket
import time
import json

sys.setrecursionlimit(40000)

TCP_IP = '192.168.0.24'
TCP_PORT = 5010  # 5001

x1_batch=[]
x2_batch=[]
y1_batch=[]
y2_batch=[]


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco
from samples.coco import coco_parse_opt as coco
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_DIR = os.path.join(ROOT_DIR, "samples/coco/logs")

# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

"""## Create Model and Load Trained Weights"""

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
COCO_MODEL_PATH = model.find_last()

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ["BG","Trash","Car","Building","Ocean","Land"]

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

"""## Run Object Detection"""
def ReceivingMsg():
    try:

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(True)
        print('wait for client...')
        conn, addr = s.accept()
        print('connected!')

        while True:
            print("receiving..")
            wid = recvall(conn, 16)
            hei = recvall(conn, 16)

            stringData = recvall(conn, int(wid) * int(hei) * 3)
            print("received!!")

            img = np.fromstring(stringData, np.uint8).reshape((int(wid), int(hei), 3))
            image = img[:, :, ::-1]

            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            idx = 0
            r = results[0]
            print("detected!!", len(r['class_ids']))
            x1_batch = []
            x2_batch = []
            y1_batch = []
            y2_batch = []
            class_batch = []
            car_pixarea = []
            for cls_id in r['class_ids']:
                if cls_id == 1 or cls_id == 2 or cls_id == 3: # trash, car, bldg
                    # if cls_id == 1 and ((int(r['rois'][idx][3])-int(r['rois'][idx][1]))*(int(r['rois'][idx][2])-int(r['rois'][idx][0])))>=2770:
                    #     idx += 1
                    #     continue
                    y1_batch.append(int(r['rois'][idx][0]))
                    x1_batch.append(int(r['rois'][idx][1]))
                    y2_batch.append(int(r['rois'][idx][2]))
                    x2_batch.append(int(r['rois'][idx][3]))
                    class_batch.append(int(r['class_ids'][idx]))

                    if cls_id == 2:
                        car_pixarea.append((int(r['rois'][idx][3])-int(r['rois'][idx][1]))*(int(r['rois'][idx][2])-int(r['rois'][idx][0])))
                idx += 1

            if not x1_batch:
                x1_batch.append(0)
                y1_batch.append(0)
                x2_batch.append(0)
                y2_batch.append(0)
                class_batch.append(0)


            print("carsize", car_pixarea)
            print("object_number", len(x1_batch))
            #print(x1_batch[0],y1_batch[0],x2_batch[0],y2_batch[0])
            dict = [x1_batch,y1_batch,x2_batch,y2_batch,class_batch]
            dict_lenv = sys.getsizeof(dict[0])

            conn.send(str(dict_lenv).encode('utf-8'))
            time.sleep(0.01)
            conn.send(json.dumps(dict[0]).encode('utf-8'))
            time.sleep(0.01)
            conn.send(json.dumps(dict[1]).encode('utf-8'))
            time.sleep(0.01)
            conn.send(json.dumps(dict[2]).encode('utf-8'))
            time.sleep(0.01)
            conn.send(json.dumps(dict[3]).encode('utf-8'))
            time.sleep(0.01)
            conn.send(json.dumps(dict[4]).encode('utf-8'))

            print("Send to Dronemap")

    except Exception:
        import traceback
        print(traceback.format_exc())


threading._start_new_thread(ReceivingMsg())