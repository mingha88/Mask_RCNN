import cv2
import skimage.io
import os
import mrcnn.ImgPreprocess as IP

# IMG_DIR = "/home/user/NAS/mingha88/PMJeju/1_kau_1010/191011/Working/"
# OUTPUT_DIR = "/home/user/NAS/mingha88/PMJeju/1_kau_1010/191011/crop_cv2/"
IMG_DIR = "/home/user/Dataset/Jeju/2_kau_1025/image/"
# OUTPUT_DIR = "/home/user/Dataset/Jeju/2_kau_1025/1024/"
OUTPUT_DIR = "/home/user/test/"

#5456, 3632
#가장 큰 사이즈는 4K(4096 x 2160)
# 1080p (1920 x 1080)  720p (1280 x 720)

crop_width = 1024
crop_height = crop_width
file_names = os.listdir(IMG_DIR)
flag = 0
for filename in file_names:
    img_cv = cv2.imread(os.path.join(IMG_DIR + filename))
    _, orientation = IP.getExif(os.path.join(IMG_DIR + filename))
    img = IP.restoreOrientation(img_cv, orientation)

    x_start = 0
    y_start = 0
    idx = 0
    for hei in range((img.shape[0] // crop_height) + 1):  # 10/13 sujung
        for wid in range((img.shape[1] // crop_width) + 1):

            if hei == (img.shape[0] // crop_height):
                y_end = (y_start + hei * crop_height) + (img.shape[0] % crop_height)
                flag = 1
            else:
                y_end = (y_start+hei*crop_height)+crop_height
                flag = 0


            if wid == (img.shape[1] // crop_width):
                x_end = (x_start + wid * crop_width) + (img.shape[1] % crop_width)
                flag = 1
            else:
                x_end = (x_start+wid*crop_width)+crop_width

            if flag == 1:
                img_crop = img[y_start + hei * crop_height:y_end, x_start + wid * crop_width:x_end]
                cv2.imwrite(OUTPUT_DIR + "{}_leftover{}_1024.jpg".format(os.path.splitext(filename)[0], idx), img_crop)

                idx += 1
            flag = 0















# for filename in file_names:
#     img_cv = cv2.imread(os.path.join(IMG_DIR+filename))
#     _, orientation = IP.getExif(os.path.join(IMG_DIR+filename))
#     img = IP.restoreOrientation(img_cv, orientation)
#
#     x_start = 0
#     y_start = 0
#     idx = 0
#     for hei in range(img.shape[0]//crop_height):
#         for wid in range(img.shape[1]//crop_width):
#
#             img_crop = img[y_start+hei*crop_height:(y_start+hei*crop_height)+crop_height, x_start+wid*crop_width:(x_start+wid*crop_width)+crop_width]
#             cv2.imwrite(OUTPUT_DIR+"{}_{}_1024.jpg".format(os.path.splitext(filename)[0],idx),img_crop)
#             idx +=1







####### Using PIL crop image############
# from PIL import Image
# import skimage.io
# import os
# import mrcnn.ImgPreprocess as IP
#
# IMG_DIR = "/home/user/NAS/mingha88/PMJeju/1_kau_1010/191011/Working/"
# OUTPUT_DIR = "/home/user/NAS/mingha88/PMJeju/1_kau_1010/191011/crop_1024/"
#
# #5456, 3632
# #가장 큰 사이즈는 4K(4096 x 2160)
# # 1080p (1920 x 1080)  720p (1280 x 720)
#
# crop_width = 1024
# crop_height = crop_width
# file_names = os.listdir(IMG_DIR)
#
# for filename in file_names:
#     # file_names = next(os.walk(IMAGE_DIR))[2]
#     img = Image.open(os.path.join(IMG_DIR+filename))
#     x_start = 0
#     y_start = 0
#     idx = 0
#     for hei in range(img.size[1]//crop_height):
#         for wid in range(img.size[0]//crop_width):
#             area = (x_start+wid*crop_width,y_start+hei*crop_height,(x_start+wid*crop_width)+crop_width,(y_start+hei*crop_height)+crop_height)
#             img_crop = img.crop(area)
#             # img_crop.show()
#             img_crop.save(OUTPUT_DIR+"{}_{}_1024.jpg".format(os.path.splitext(filename)[0],idx))
#             idx +=1
#
#
# test =1
# # img = img.resize((1024, 1024), Image.ANTIALIAS)
# # img.save("/home/user/resize.jpg")


