import cv2
import numpy as np

# ---Charges the image to analyze ---
image = cv2.imread("bills_total.jpg")
if image is None:
    print("Image not found")
    exit()