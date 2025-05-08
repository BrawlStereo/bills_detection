import cv2
import numpy as np

# ---Charges the image to analyze ---
image = cv2.imread("bills_total.jpg")
if image is None:
    print("Image not found")
    exit()

# Converts the image to  grayscale so binarization accepts it
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applies adaptative binarization to the image so the original white background is black and the bills in white tones
#which facilitates the bills contorus detection
image_binary = cv2.adaptiveThreshold(
    image_gray, 
    255,                                #max value taken by pixels less than the threshold
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     #calculate threshold with gaussian weighted sum
    cv2.THRESH_BINARY_INV, 
    11,                                 #size of pixel surrounding
    2                                   #constant to add to gaussian result
)