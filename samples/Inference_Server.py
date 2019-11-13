

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

sys.setrecursionlimit(40000)
TCP_IP = '192.168.0.24'
TCP_PORT = 5001  # 5001



# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco
from samples.coco import coco_parse_opt as coco
#%matplotlib inline

# Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = "/home/user/Work/Mask_RCNN/samples/coco/logs" #os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH ="/home/user/Work/Mask_RCNN/samples/coco/logs/coco20191105T1802/mask_rcnn_coco_0159.h5" #Checkpoint Path: /home/user/Work/Mask_RCNN/samples/coco/logs/coco20191105T1724/mask_rcnn_coco_{epoch:04d}.h5
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# IMAGE_DIR = "/home/user/Dataset/Jeju/annotated"#os.path.join(ROOT_DIR, "images")

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

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ["0","1","2","3","4","5"]

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
        #toliveDronmap = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #toliveDronmap.connect((TCP_IP2, TCP_PORT2))

        while True:
            print("receiving..")
            wid = recvall(conn, 16)
            hei = recvall(conn, 16)

            stringData = recvall(conn, int(wid) * int(hei) * 3)
            print("received!!")
            all_imgs = []

            classes = {}

            bbox_threshold = 0.1

            visualise = True

            st = time.time()
            # img = cv2.imread(filepath)
            image = np.fromstring(stringData, np.uint8).reshape((int(wid), int(hei), 3))
            #PIL convert and crop
            im = Image.fromarray(np.uint8(cm.gist_earth(myarray) * 255))
            ##PIL image crop
            # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
            test = 1
            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])

# Load a random image from the images folder
for i in range(10):
    file_names = next(os.walk(IMAGE_DIR))[2]

    # fig.savefig('demo.png', bbox_inches='tight')

    test = 0



threading._start_new_thread(ReceivingMsg())