import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image
import torch
import os
import random
import math
from PIL import ImageFilter

class GaussianBlur(object):
	"""Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

	def __init__(self, sigma=[.1, 2.]):
		self.sigma = sigma

	def __call__(self, x):
		sigma = random.uniform(self.sigma[0], self.sigma[1])
		x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
		return x

def get_cifar_train_augmentations(type):
	minimum_crop = 0.2
	if type == 'normal':
		train_transforms = transforms.Compose([
			transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
			transforms.RandomApply([
					transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
			], p=0.8),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	elif type == 'minusblur':
		train_transforms = transforms.Compose([
			transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
			transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
			transforms.RandomGrayscale(p=0.2),
			# RandomBlur(),
			#transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	elif type == 'plusrotation':
		train_transforms = transforms.Compose([
			transforms.RandomApply([
					transforms.RandomRotation(30, expand=False),
			], p=0.5),
			transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
			transforms.RandomApply([
					transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
			], p=0.8),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	return train_transforms
	
def make_grid(pils, cols, padding=5):
	rows = int(math.ceil(len(pils)/cols))
	img_w, img_h = pils[0].size
	background = Image.new('RGB',((img_w+padding)*cols-padding, (img_h+padding)*rows-padding), (255, 255, 255))
	for i, img in enumerate(pils):
		i_col = i % cols
		i_row = i // rows
		offset = (i_col*(img_w+padding), i_row*(img_h+padding))
		background.paste(img, offset)
	return background

if __name__ == '__main__':

	path = os.path.expanduser('~/lenna.png')
	img = Image.open(path)

	transform = transforms.Compose([
		# transforms.RandomResizedCrop(size=224, scale=(0.2,1.)),
		transforms.RandomRotation(30, resample=Image.BICUBIC, expand=False),
		transforms.RandomResizedCrop(size=224, scale=(0.2,1.)),
	])

	train_transforms = transforms.Compose([
				transforms.RandomApply([
					transforms.RandomRotation(30, expand=False),  # not strengthened
				], p=0.5),
				transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
				transforms.RandomApply([
					transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
				], p=0.8),
				
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.RandomHorizontalFlip(),
				# transforms.ToTensor(),
				# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	imgs = []
	for i in range(100):
		new_img = train_transforms(img)
		imgs.append(new_img)
	background = make_grid(imgs, 10)
	background.show()