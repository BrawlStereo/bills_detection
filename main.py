import cv2
import numpy as np
import easyocr

# Charges the image to analyze
image = cv2.imread("bills_total.jpg")
if image is None:
    print("Image not found")
    exit()
image_original = image.copy() #copy in which the OCR analisis will be made

# Converts the image to grayscale so binarization accepts it
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

# Applies morphology to close internal wholes within the bills without joining them with each other
kernel = np.ones((5, 5), np.uint8)
image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)

# Finds and draws in red the bills contours
contours, _ = cv2.findContours(
    image_binary, 
    cv2.RETR_EXTERNAL,          # just picks the bills' external contours
    cv2.CHAIN_APPROX_SIMPLE     #Simplify contour points to avoid repeated points
)
cv2.drawContours(
    image, 
    contours,      #list of arrays with contours coordenates 
    -1,            #draw all contours
    (0, 0, 255),   #red contours
    1              #line width of 1
)                  

# Easyocr initialization to spanish because the bills are mexican
reader = easyocr.Reader(['es'], gpu=False)

number_bills_counter = {}  # dictionary to count bills of each type
total = 0    # total sum of the money shown

# Process every contour 
for i, contour in enumerate(contours):

    #Ignores contours that aren't bills
    area = cv2.contourArea(contour)
    if area < 15000:
        continue

    # Cuts the image of the bill from the unmodified image
    x, y, w, h = cv2.boundingRect(contour)      #gets contour dimensions, x is initial position in x, y is the initial position in y,
                                            #w the bill width and h the bill height
    bill = image_original[y:y+h, x:x+w]  #gets the portion of the image since line y until y+h and since column x until x+w

    # Applies gussian blur to eliminate small details that ocr could detect
    bill_blurred = cv2.GaussianBlur(bill, (5, 5), 0)

    # Apply OCR
    results = reader.readtext(bill_blurred)
    #print(f"OCR en bill {i+1} (con blur):")

    detected_value = None #bill's value detected by ocr

    # Look if the text detected by ocr is a valid bill value
    for _, text, _ in results:
        #text = text.replace("$", "").replace(",", "").strip()     #deletes possible unnecesary symbols detected
        print("→ Detected:", text)
        text = text.lower().replace("o", "0").replace("s", "5").replace("l", "1") #replaces posibble confussions between a char and int
        for possible_value in [1000, 500, 200, 100, 50, 20]:
            if str(possible_value) in text:
                detected_value = possible_value
                break
        if detected_value:
            break
   
    

