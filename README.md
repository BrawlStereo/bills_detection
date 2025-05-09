# Bills detection

This project applies a sequence of computer vision techniques to detect Mexican bills in an image and calculate 
the total amount of money displayed. It uses adaptive thresholding to detect the bills' contours, morphological 
operations to close internal gaps, Gaussian blur to reduce noise and emphasize the printed value, and EasyOCR 
to read the number and compute the total.

## Requirements

Install the following python libraries:

- OpenCV:
```
pip install opencv-python
```

- Numpy:
```
pip install numpy
```

- EasyOCR:
```
pip install easyocr
```

- Matplotlib:
```
pip install matplotlib
```

## How It Works

- When prompted, enter the filename of the image you want to analyze. Available options include:

```
bills_case_1.jpg
```

```
bills_case_2.jpg
```

```
bills_total.jpg
```

- The total amount of money and the breakdown of bills will be printed in the console.

- A new image named result.jpg will be saved, where each detected bill is highlighted with a green rectangle
and its value is annotated on top.

## Execution

Run the script from your terminal using:

```
python -u "path in which you downloaded the repository/main.py" 
```