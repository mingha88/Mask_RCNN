import cv2
from PIL import Image
import numpy as np
from PIL.ExifTags import TAGS, GPSTAGS

def getExif(path):
    src_image = Image.open(path)
    info = src_image._getexif()
    test = 1
    # if info is not None:
    #     # Focal Length
    #     # focalLength = info[37386]
    #     # focal_length = focalLength[0] / focalLength[1] # unit: mm
    #     # focal_length = focal_length * pow(10, -3) # unit: m
    #
    #     # Orientation
    #     orientation = info[274]
    # else:
    #     orientation = None
    try:
        orientation = info[274]
    except:
        orientation = 0

    # return focal_length, orientation
    return orientation

def restoreOrientation(image, orientation):
    if orientation == 8:
        restored_image = rotate(image, -90)
    elif orientation == 6:
        restored_image = rotate(image, 90)
    elif orientation == 3:
        restored_image = rotate(image, 180)
    else:
        restored_image = image

    return restored_image

def rotate(image, angle):
    # https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    height = image.shape[0]
    width = image.shape[1]
    center = (width/2, height/2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # compute the new bounding dimensions of the image
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # adjust the rotation matrix to take into account translation
    rotation_mat[0, 2] += bound_w / 2 - center[0]
    rotation_mat[1, 2] += bound_h / 2 - center[1]

    # perform the actual rotation and return the image
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def restoreVertices(bbox,ori): #dict = [x1_batch,y1_batch,x2_batch,y2_batch,class_batch]

    X1_batch = []
    Y1_batch = []
    X2_batch = []
    Y2_batch = []
    degree ={6: 90, 3: 180}

    cos = np.cos(degree[ori] * np.pi/180)
    sin = np.sin(degree[ori] * np.pi/180)

    for i in range(len(bbox[0])):
        x1 = bbox[0][i]
        y1 = bbox[1][i]
        x3 = bbox[2][i]
        y3 = bbox[3][i]

        X1 = x1 * cos + y1 * sin
        X3 = x3 * cos + y3 * sin
        Y1 = -x1 * sin + y1 * cos
        Y3 = -x3 * sin + y3 * cos
        ## origin shift
        X1 += abs(X3 - X1)
        X3 += abs(X3 - X1)

        if degree[ori] == 180:
            Y1 += abs(Y3-Y1)
            Y3 += abs(Y3-Y1)

        X1_batch.append(int(X1))
        X2_batch.append(int(X3))
        Y1_batch.append(int(Y1))
        Y2_batch.append(int(Y3))

    return [X1_batch,Y1_batch,X2_batch,Y2_batch,bbox[4]]









