import os
import cv2
import argparse

def max(a,b):return a if a>b else b
parser = argparse.ArgumentParser()
parser.add_argument('source', default='', type=str)

args = parser.parse_args()

img = cv2.imread(args.source)
h, w = img.shape[0], img.shape[1]
img = cv2.resize(img, (max(h, w), max(h, w)))
cv2.imwrite(args.source, img)