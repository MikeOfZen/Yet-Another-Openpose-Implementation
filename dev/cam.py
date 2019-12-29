import tensorflow as tf
import cv2
import numpy as np

import post_processing.post as post
import matplotlib.pyplot as plt

model_path = "../trained_models/model11_test-15Sun1219-2101"
model = tf.keras.models.load_model(model_path)

import configs.keypoints_config as kc
import configs.post_config as post_config

post.Skeletonizer.config(kc.KEYPOINTS_DEF, kc.JOINTS_DEF, post_config)
post.Skeleton.config(kc.KEYPOINTS_DEF, kc.JOINTS_DEF)


def process2(img):
    # f=tf.io.read_file(r"C:\Users\flash\Desktop\tmp\img.JPG")
    # img=tf.image.decode_jpeg(f)
    img = tf.image.resize(img, (360, 360))
    input_img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    input_img /= 255
    input_img = input_img[tf.newaxis, ...]
    pafs, kpts = model.predict(input_img)
    pafs = pafs[0]
    kpts = kpts[0]
    skeletonizer = post.Skeletonizer(kpts, pafs)
    skeletons = skeletonizer.create_skeletons()

    output_img = np.zeros((46, 46, 3))
    for skeleton in skeletons:
        skeleton.draw_skeleton(output_img)
    output_img = cv2.resize(output_img, (500, 500))
    return output_img


# initialize the camera
cam = cv2.VideoCapture(0)  # 0 -> index of camera
s, cam_img = cam.read()
if s:  # frame captured without any errors
    cv2.namedWindow("cam-test", cv2.WINDOW_AUTOSIZE)
    while True:
        s, cam_img = cam.read()
        processed_img = process2(cam_img)
        cv2.imshow("cam-test", processed_img)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to stop
            break
    cv2.destroyWindow("cam-test")
