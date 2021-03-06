from __future__ import division
from __future__ import print_function
from PIL import Image
from PIL import ImageFilter
import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
import datasets.list as listData

class RandomBlur(object):
	def __call__(self, pic):
		p = np.random.rand()
		r = np.random.randint(1, 8)
		if p < 0.5:
			return pic.filter(ImageFilter.GaussianBlur(radius=r))
		else:
			return pic

	def __repr__(self):
		return 'Blur'

class GaussianBlur(object):
	"""Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

	def __init__(self, sigma=[.1, 2.]):
		self.sigma = sigma

	def __call__(self, x):
		sigma = random.uniform(self.sigma[0], self.sigma[1])
		x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
		return x


def get_linear_dataloader(args):
	train_transforms = transforms.Compose([
		# transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
		# transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
		# transforms.RandomGrayscale(p=0.2),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	val_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms) if args.dataset == 'cifar10' else datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)

	val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transforms) if args.dataset == 'cifar10' else datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transforms)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, pin_memory=False,
						num_workers=args.n_workers)	
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
						shuffle=False, pin_memory=False,
						num_workers=args.n_workers)
	return train_loader, val_loader

def get_dataloader(args, train_transforms):
	val_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	
	train_dataset = CIFAR10Instance(root='./data', train=True, download=True, transform=train_transforms) if args.dataset == 'cifar10' else CIFAR100Instance(root='./data', train=True, download=True, transform=train_transforms)

	val_dataset = CIFAR10Instance(root='./data', train=False, download=True, transform=val_transforms) if args.dataset == 'cifar10' else CIFAR100Instance(root='./data', train=False, download=True, transform=val_transforms)

	train_ordered_labels = np.array(train_dataset.targets)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, pin_memory=False,
						num_workers=args.n_workers)	
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
						shuffle=False, pin_memory=False,
						num_workers=args.n_workers)
	return train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset
	

class CIFAR10Instance(datasets.CIFAR10):
	"""CIFAR10Instance Dataset.
	"""
	def __getitem__(self, index):
		if self.train:
			img, target = self.data[index], self.targets[index]
		else:
			img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img1 = self.transform(img)
			img2 = self.transform(img)

		return img1, img2, target

class CIFAR100Instance(CIFAR10Instance):
	"""CIFAR100Instance Dataset.

	This is a subclass of the `CIFAR10Instance` Dataset.
	"""
	base_folder = 'cifar-100-python'
	url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
	filename = "cifar-100-python.tar.gz"
	tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
	train_list = [
		['train', '16019d7e3df5f24257cddd939b257f8d'],
	]

	test_list = [
		['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
	]
	meta = {
		'filename': 'meta',
		'key': 'fine_label_names',
		'md5': '7973b15100ade9c7d40fb424638fde48',
	}

