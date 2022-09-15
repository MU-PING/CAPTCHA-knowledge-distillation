import cv2
import numpy as np

# Training data
train_data = np.stack([cv2.imread("Dataset\\Training\\Training-CAPTCHA\\"+ "{:>06d}.jpg".format(i))for i in range(50000)], axis=0)

# Inference data
inference_data = np.stack([cv2.imread("Dataset\\Inference\\Inference-CAPTCHA\\"+ "{:>06d}.jpg".format(i))for i in range(10000)], axis=0)

temperature = 3.25