import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source', nargs="+")
parser.add_argument('--dest', type=str)
parser.add_argument('--w', type=int)
parser.add_argument('--h', type=int)

args = parser.parse_args()

imgs = []
w = 10
for ipath in args.source:
	img = cv2.imread(ipath)
	img = cv2.resize(img, (args.w, args.h))
	imgs.append(img)
	imgs.append(np.zeros(shape=(img.shape[0], w, 3)))
imgs = np.concatenate(imgs[:-1], axis=1)

cv2.imwrite(args.dest, imgs)
