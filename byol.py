from __future__ import division
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
import datasets.byol_cifar as cifar
import datasets.svhn as svhn
import datasets.imagenet as imagenet
from objective import l2_normalize
from models.resnet_cifar import ResNet18 as ResNet18_cifar
from models.resnet_cifar import ResNet50 as ResNet50_cifar
from models.resnet import resnet18, resnet50
# from models.shufflenetv2 import shufflenet_v2_x1_0
from torch.utils.data import DataLoader
from utils import *
import objective
import logging
import torch.distributions.beta as beta
from models.wideresnet import WideResNetInstance
import augmentation

class Recording(object):
	def __init__(self, names):
		self.names = names
		self.epoch_meters = []
		self.meters = []
		for i in range(len(names)):
			self.meters.append(AvgMeter())
			self.epoch_meters.append([])

	def add(self, variables):
		for i, variable in enumerate(variables):
			self.meters[i].add(variable)

	def add_scalar(self, writer, current_epoch):
		for i, name in enumerate(self.names):
			writer.add_scalar(name, self.meters[i].get(), current_epoch)
		logging.info(str(self))

	def save(self, exp):
		for i, name in enumerate(self.names):
			np.save(os.path.join(exp, '{}.npy'.format(name)), self.epoch_meters[i])

	def epoch_record(self):
		for i, name in enumerate(self.names):
			self.epoch_meters[i].append(self.meters[i].get())
			self.meters[i].clear()

	def __str__(self):
		descriptor = ''
		if self.names == []:
			return descriptor

		for i, name in enumerate(self.names):
			descriptor += '{}:{:.4f},'.format(name, self.meters[i].get())
		return descriptor

class BYOL(object):
	def __init__(self, args):
		cudnn.benchmark = True
		self.args = args
		self.m = 0.996
		self.create_experiment()
		self.get_logger()
		self.device_ids = list(map(lambda x: int(x), args.gpus.split(',')))
		self.device = torch.device('cuda: 0')
		self.get_network()
		self.get_dataloader()
		self.get_optimizer()
		logging.info(beautify(args))
		self.start_epoch = 0
		if args.resume_path != '':
			try:
				self.load_checkpoint(args.resume_path)
			except:
				pass
		self.all_record_similarities = []
		self.all_record_indices = []
		self.main_loop()

	def create_experiment(self):
		args = self.args
		if not os.path.exists(args.exp):
			os.makedirs(args.exp)
		if not os.path.exists(os.path.join(args.exp, 'models')):
			os.makedirs(os.path.join(args.exp, 'models'))
		if not os.path.exists(os.path.join(args.exp, 'similarities')):
			os.makedirs(os.path.join(args.exp, 'similarities'))

	def get_logger(self):
		args = self.args
		if not os.path.exists(os.path.join(args.exp, 'logs')):
			os.makedirs(os.path.join(args.exp, 'logs'))
		if self.args.embedding:
			if not os.path.exists(os.path.join(args.exp, 'runs_'+self.args.embedding_layer)):
				os.makedirs(os.path.join(args.exp, 'runs_'+self.args.embedding_layer))
			self.writer = SummaryWriter(logdir=os.path.join(self.args.exp, 'runs_'+self.args.embedding_layer))
		else:
			if not os.path.exists(os.path.join(args.exp, 'runs')):
				os.makedirs(os.path.join(args.exp, 'runs'))
			self.writer = SummaryWriter(logdir=os.path.join(self.args.exp, 'runs'))
		self.logger = getLogger(args.exp)

	def get_dataloader(self):
		args = self.args
		if args.dataset.startswith('cifar'):
			if args.rotation:
				train_transforms = augmentation.get_cifar_train_augmentations(type='plusrotation')
			else:
				train_transforms = augmentation.get_cifar_train_augmentations(type='normal')
			train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset = cifar.get_dataloader(args, train_transforms) 
		elif args.dataset.startswith('imagenet'):
			train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset = imagenet.get_instance_dataloader(args)
		elif args.dataset == 'svhn':
			train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset = svhn.get_dataloader(args)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.train_ordered_labels = train_ordered_labels
		
	def get_network(self):
		args = self.args
		if args.network == 'resnet18_cifar':
			network = ResNet18_cifar(256, dropout=args.dropout, non_linear_head=True, mlpbn=args.mlpbn, bnaffine=self.args.bnaffine)
			target_network = ResNet18_cifar(256, dropout=args.dropout, non_linear_head=True, mlpbn=args.targetmlpbn, bnaffine=self.args.bnaffine)
			noise_network = ResNet18_cifar(256, dropout=args.dropout, non_linear_head=True, mlpbn=args.targetmlpbn, bnaffine=self.args.bnaffine)
		elif args.network == 'resnet50_cifar':
			network = ResNet50_cifar(256, dropout=args.dropout, mlpbn=args.mlpbn, non_linear_head=True, bnaffine=self.args.bnaffine)
			target_network = ResNet50_cifar(256, dropout=args.dropout, mlpbn=args.targetmlpbn, non_linear_head=True, bnaffine=self.args.bnaffine)
		elif args.network == 'resnet18':
			network = resnet18(non_linear_head=True, mlpbn=args.mlpbn, bnaffine=self.args.bnaffine)
			target_network = resnet18(non_linear_head=True, mlpbn=args.targetmlpbn, bnaffine=self.args.bnaffine)
		elif args.network == 'resnet50':
			network = resnet50(non_linear_head=True, mlpbn=args.mlpbn, bnaffine=self.args.bnaffine)
			target_network = resnet50(non_linear_head=True, mlpbn=args.targetmlpbn, bnaffine=self.args.bnaffine)
		self.network = nn.DataParallel(network, device_ids=self.device_ids)
		self.network.to(self.device)
		self.target_network = nn.DataParallel(target_network, device_ids=self.device_ids)
		self.target_network.to(self.device)
		self.noise_network = nn.DataParallel(noise_network, device_ids=self.device_ids)
		self.noise_network.to(self.device)
		if self.args.prebn:
			self.predictor = nn.Sequential(
							nn.Linear(256, 256),
							nn.BatchNorm1d(256, affine=self.args.bnaffine),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
			)
		else:
			self.predictor = nn.Sequential(
							nn.Linear(256, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
			)
		self.predictor = nn.DataParallel(self.predictor, device_ids=self.device_ids)
		self.predictor.to(self.device)

	def _update_param(self):
		with torch.no_grad():
			total_operation = 0
			for online_param, target_param in zip(self.network.named_parameters(), self.target_network.named_parameters()):
				if online_param[0] == target_param[0]:
					target_param[1].data = target_param[1].data * self.m + online_param[1].data * (1. - self.m)
					# print('update from {}->{}'.format(online_param[0], target_param[0]))
					total_operation += 1

			# transfer the last linear layer
			last_i = 0
			for online_param, target_param in zip(list(self.network.named_parameters())[::-1], list(self.target_network.named_parameters())[::-1]):
				target_param[1].data = target_param[1].data * self.m + online_param[1].data * (1. - self.m)
				# print('update from {}->{}'.format(online_param[0], target_param[0]))
				total_operation += 1
				last_i += 1
				if last_i == 2:
					break
			assert total_operation == min( len(list(self.target_network.parameters())), len(list(self.network.parameters())) )

	def update_param(self):
		with torch.no_grad():
			for online_param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
				target_param.data = target_param.data * self.m + online_param.data * (1. - self.m)
	
	def add_noise(self):
		with torch.no_grad():
			for target_param, noise_param in zip(self.target_network.parameters(), self.noise_network.parameters()):
				magnitude = target_param.data.abs().mean()
				torch.nn.init.normal_(noise_param, mean=0.0, std=max(magnitude*self.args.std, 1e-7))

			for target_param, noise_param in zip(self.target_network.parameters(), self.noise_network.parameters()):
				target_param.data = target_param.data + noise_param.data

	def remove_noise(self):
		with torch.no_grad():
			for target_param, noise_param in zip(self.target_network.parameters(), self.noise_network.parameters()):
				target_param.data = target_param.data - noise_param.data

	def get_optimizer(self):
		args = self.args
		if args.network == 'pre-resnet18' or args.network == 'pre-resnet50':
			logging.info('Exclude bns from weight decay, copied from LocalAggregation proposed by Zhuang et al [ICCV 2019]')
			parameters = exclude_bn_weight_bias_from_weight_decay(self.network, weight_decay=args.weight_decay)
		else:
			parameters = self.network.parameters()

		self.optimizer = torch.optim.SGD(
			list(parameters) + list(self.predictor.parameters()),
			lr=args.lr,
			momentum=0.9,
			weight_decay=args.weight_decay,
		)


	def load_checkpoint(self, path):
		args = self.args
		logging.info('resume from {}'.format(args.resume_path))
		checkpoint = torch.load(args.resume_path)
		self.network.load_state_dict(checkpoint['state_dict'])
		self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
		self.target_network.load_state_dict(checkpoint['target_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.start_epoch = checkpoint['epoch']

	def main_loop(self):
		logging.info('start training')
		best_acc = 0.0
		args = self.args
		all_accs = []
		if not self.args.embedding:
			try:
				for i_epoch in range(self.start_epoch, args.max_epoch):
					self.current_epoch = i_epoch
					adjust_learning_rate(args.lr, args.lr_decay_steps, self.optimizer, i_epoch, lr_decay_rate=args.lr_decay_rate, cos=args.cos, max_epoch=args.max_epoch)
					self.train()

					save_name = 'checkpoint.pth'
					checkpoint = {
						'epoch': i_epoch + 1,
						'state_dict': self.network.state_dict(),
						'optimizer': self.optimizer.state_dict(),
						'target_state_dict': self.target_network.state_dict(),
						'predictor_state_dict': self.predictor.state_dict(),
					}
					torch.save(checkpoint, os.path.join(args.exp, 'models', save_name))

					# scheduler.step()
					# validate(network, memory_bank, val_loader, train_ordered_labels, device)
					if i_epoch in [30, 60, 120, 160, 200, 400, 600]:
						torch.save(checkpoint, os.path.join(args.exp, 'models', '{}.pth'.format(i_epoch+1)))

					# cluster
				
			except KeyboardInterrupt as e:
				logging.info('KeyboardInterrupt at {} Epochs'.format(i_epoch))
				save_result(self.args)
				exit()
		else:
			self.inference()

		# self.recording.save(self.args.exp)
		save_result(self.args)

	def record_similarity(self, similarities, index):
		p = self.args.record_prob
		if np.random.uniform(low=0.0, high=1.0) < p:
			self.all_record_similarities.append(similarities)
			self.all_record_indices.append(index)

	def save_record_similarity(self):
		if self.all_record_indices == []:
			return
		self.all_record_similarities = torch.cat(self.all_record_similarities, dim=0)
		self.all_record_indices = torch.cat(self.all_record_indices, dim=0)
		if self.current_epoch % self.args.record_freq == 0:
			torch.save(
				{
					'sim': self.all_record_similarities,
					'index': self.all_record_indices,
				},
				os.path.join(self.args.exp, 'similarities', 'sim_epoch{}.pth'.format(self.current_epoch))
			)
		self.all_record_similarities = []
		self.all_record_indices = []

	def train(self):
		self.network.train()
		losses = AvgMeter()
		all_weights = []
		pbar = tqdm(self.train_loader)

		for data in pbar:
			img1 = data[0].to(self.device)
			img2 = data[1].to(self.device)

			featA1 = self.network(img1)
			if self.args.noise:
				self.add_noise()
			featA2 = l2_normalize(self.target_network(img2)).to(self.device)

			predicted_featA2 = l2_normalize(self.predictor(featA1)).to(self.device)
			loss1 = ((predicted_featA2 - featA2)**2).sum(dim=1).mean()

			featB1 = self.network(img2)
			featB2 = l2_normalize(self.target_network(img1)).to(self.device)
			if self.args.noise:
				self.remove_noise()
			predicted_featB2 = l2_normalize(self.predictor(featB1)).to(self.device)
			loss2 = ((predicted_featB2 - featB2)**2).sum(dim=1).mean()

			loss = loss1 + loss2

			losses.add(loss.item())

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if self.args.mlpbn == self.args.targetmlpbn:
				self.update_param()
			else:
				self._update_param()

			lr = self.optimizer.param_groups[0]['lr']
			pbar.set_description("Epoch:{} [lr:{}]".format(self.current_epoch, lr))
			info = 'L: {:.4f}'.format(losses.get())
			pbar.set_postfix(info=info)

		self.writer.add_scalar('L', losses.get(), self.current_epoch)
		# self.recording.add_scalar(self.writer, self.current_epoch)
		# self.recording.epoch_record()
		logging.info('Epoch {}: L: {:.4f}'.format(self.current_epoch, losses.get()))
		# self.save_record_similarity()

	@torch.no_grad()
	def inference(self):
		self.network.eval()
		if self.args.embedding_layer == 'fc':
			points = torch.zeros(len(self.val_dataset), 256).to(self.device).detach()
		elif self.args.embedding_layer == 'lastconv':
			points = torch.zeros(len(self.val_dataset), 512).to(self.device).detach()
		labels = []
		pointer = 0
		pbar = tqdm(self.val_loader)
		for data in pbar:
			img = data[0].to(self.device)
			targets = data[2]
			bs = img.size(0)
			if self.args.embedding_layer == 'fc':
				feat = self.network(img, layer=7)
			elif self.args.embedding_layer == 'lastconv':
				feat = self.network(img, layer=6)
			points[pointer:pointer+bs] = feat
			targets = list(map(lambda a: a.item(), targets))
			labels.extend(targets)
			pointer = pointer + bs
		
		self.writer.add_embedding(points, metadata=labels)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--stage', default='train', type=str)

	parser.add_argument('--data', default='', type=str)
	parser.add_argument('--cudaenv', default='', type=str)
	parser.add_argument('--gpus', default='0,1,2,3', type=str)
	parser.add_argument('--max_epoch', default=201, type=int)
	parser.add_argument('--lr_decay_steps', default='160,190,200', type=str)
	parser.add_argument('--exp', default='', type=str)
	parser.add_argument('--resume_path', default='', type=str)
	parser.add_argument('--res_path', default='results.csv', type=str)

	parser.add_argument('--dataset', default='imagenet', type=str)
	parser.add_argument('--lr', default=0.03, type=float)
	parser.add_argument('--lr_decay_rate', default=0.1, type=float)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--weight_decay', default=5e-4, type=float)
	parser.add_argument('--n_workers', default=32, type=int)
	parser.add_argument('--dropout', action='store_true')
	parser.add_argument('--rotation', action='store_true')
	parser.add_argument('--cos', action='store_true')
	parser.add_argument('--mlpbn', default=1, type=int)
	parser.add_argument('--targetmlpbn', default=1, type=int)
	parser.add_argument('--prebn', default=1, type=int)
	parser.add_argument('--bnaffine', default=1, type=int)
	parser.add_argument('--std', default=0.01, type=float)
	parser.add_argument('--noise', default=1, type=int)

	parser.add_argument('--network', default='resnet18', type=str)
	parser.add_argument('--record_prob', default=0.1, type=float)
	parser.add_argument('--record_freq', default=10, type=float)

	parser.add_argument('--embedding', default=0, type=int)
	parser.add_argument('--embedding_layer', default='fc', type=str)
	# exclusive best to be 0
	args = parser.parse_args()
	Runner = BYOL(args)
	return args


def run_eval_linear(args):
	lr = 10.0 if args.network.startswith('imagenet') else 30.0
	cmd = "CUDA_VISIBLE_DEVICES={} python -m downstream.eval_linear --learning_rate {} --model {} --save_folder '{}' --model_path '{}' --dataset {} --gpu 0 --data_folder {}".format(args.cudaenv, lr, args.network, os.path.join(args.exp, 'linear'), os.path.join(args.exp, 'models', 'checkpoint.pth'), args.dataset, args.data)
	logging.info(cmd)
	os.system(cmd)

def run_semi_supervised(args):
	semi_fraction = [250, 500, 1000, 2000, 4000]
	for fraction in semi_fraction:
		cmd = "python -m downstream.semi_supervised --dataset {} --gpus 0 --exp '{}' --list '{}' --pretrain_path '{}' --network {} --data_folder {}".format(args.dataset, os.path.join(args.exp, 'semi_{}'.format(fraction)), 'datasets/lists/cifar_{}.txt'.format(fraction), os.path.join(args.exp, 'models', 'best.pth'), args.network, args.data)
		logging.info(cmd)
		os.system(cmd)


if __name__ == '__main__':
	args = main()
	run_eval_linear(args)
	#run_semi_supervised(args)


#python byol.py --data './data/' --cudaenv '0,1,2,3' --gpus '0,1,2,3' --exp '/matrix/BYOL/res50/' --dataset 'cifar100' --batch_size 256 --lr 0.06 --network 'resnet50_cifar'
