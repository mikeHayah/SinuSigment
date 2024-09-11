import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
import torch.nn as nn
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		#self.criterion2 = self.diceLoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()
		
		# addtional 
		self.BottleNeck_size = 1024
	
	def diceLoss(self, x, y):
		intersection = torch.sum(x * y) + 1e-7
		union = torch.sum(x) + torch.sum(y) + 1e-7
		return 1 - 2 * intersection / union
		
			
	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=1,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=1,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)
		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img
	
	
	def binning(self, x):
		batch_size, channels, height, width = x.size()
		x = x.view(batch_size, channels, height // 2, 2, width // 2, 2)
		x = x.float()
		x = x.mean(dim=(3, 5))
		return x
	
	def bin_gt(self, gt):
		for i in range(4):
			gt = self.binning(gt)
		return gt
		
	def check_bin(self, bn_gt, original_gt, i):
		# create the output path
		output_path = './dataset/binned_mask/'
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		stretching = nn.Upsample(scale_factor=16)	
		mask_stretched = stretching(bn_gt)
		for n in range(mask_stretched.shape[0]):
			mask = mask_stretched[n, 0, :, :]
			mask = mask.squeeze(0)
			mask = mask.squeeze(0)
			mask = mask - mask.min()
			mask = mask / mask.max()
			mask = ((mask) * 255).byte()
			mask = mask.cpu().numpy()
			mask = Image.fromarray(mask, mode='L')
			k = i*64 + n
			mask.save(os.path.join(output_path,"maskgt_"+str(k)+".tif"))
			gt = original_gt[n, 0, :, :]
			gt = gt.squeeze(0)
			gt = gt.squeeze(0)
			gt = gt - gt.min()
			gt = gt / gt.max()
			gt = ((gt) * 255).byte()
			gt = gt.cpu().numpy()
			gt = Image.fromarray(gt, mode='L')
			k = i*64 + n
			gt.save(os.path.join(output_path,"gt_"+str(k)+".tif"))

	def check_images(self, images, GT):
		
		# create the output path
		output_path = './dataset/check_images/'
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		images_path = './dataset/check_images/images/'
		if not os.path.exists(images_path):
			os.makedirs(images_path)
		gt_path = './dataset/check_images/gt/'
		if not os.path.exists(gt_path):
			os.makedirs(gt_path)	
		to_pil = ToPILImage()	
		for n in range(images.shape[0]):
			image = images[n, 0, :, :]
			image = image.squeeze(0)
			#image = image.squeeze(0)
			image = image - image.min()
			image = image / image.max()
			image = ((image) * 255).byte()
			#image = image.cpu().numpy()
			#image = Image.fromarray(image, mode='L')
			image = to_pil(image)
			image.save(os.path.join(images_path,"image_"+str(n)+".png"))
			mask = GT[n, 0, :, :]
			mask = mask.squeeze(0)
			#mask = mask.squeeze(0)
			mask = mask - mask.min()
			mask = mask / mask.max()
			mask = ((mask) * 255).byte()
			#mask = mask.cpu().numpy()
			#mask = Image.fromarray(mask, mode='L')
			mask = to_pil(mask)
			mask.save(os.path.join(gt_path,"gt_"+str(n)+".png"))
			
	def visualiz_processing(self, images, GTs, SR_probs, BN_img, fig, axes):
		 
		for i in range(3):
			image = images[i].squeeze(0).byte().cpu().numpy()
			#image[image>0] = 255
			axes[i,0].clear()
			axes[i,0].imshow(image)
			axes[i,0].set_title("image")
			axes[i,0].axis('off')
			axes[i,0].set_aspect('equal')
			
		for i in range(3):
			GT = GTs[i].squeeze(0).byte().cpu().numpy()
			GT[GT>0.5] = 255
			axes[i,1].clear()
			axes[i,1].imshow(GT)
			axes[i,1].set_title("GT")
			axes[i,1].axis('off')
			axes[i,1].set_aspect('equal')
			
		for i in range(3): 
			SR_prob = SR_probs[i].detach().cpu().numpy()	#.byte().cpu().numpy()
			SR_prob[SR_prob>=0.5] = 255
			axes[i,2].clear()
			axes[i,2].imshow(SR_prob)
			axes[i,2].set_title("Prediction")
			axes[i,2].axis('off')
			axes[i,2].set_aspect('equal')
			

		for i in range(3):
			BN = BN_img[i].detach().cpu().numpy()
			BN = BN * 255
			axes[i,3].clear()
			axes[i,3].imshow(BN)
			axes[i,3].set_title("BottleNeck")
			axes[i,3].axis('off')
			axes[i,3].set_aspect('equal')
	
		plt.pause(0.001)
		plt.draw()
                    
		

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			startTime = time.time()
			lr = self.lr
			best_unet_score = 0.
			best_unet = None
			best_epoch = 0
			
			plt.ion()
			fig, axes = plt.subplots(3, 4, figsize=(25,25)) 
			
			for epoch in tqdm(range(self.num_epochs)):
			#for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				val_loss = 0
				test_loss = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0
				
				print("epoch", epoch+1)
				# check data loader
				#images_patches, GT_patches = self.train_loader.dataset.__getitem__(60)
				
				for i, (images_patches, GT_patches) in enumerate(self.train_loader):
					# train for each patch
					torch.save(self.unet.state_dict(), unet_path)
					
					for i in range(images_patches.size(1)):
						images = images_patches[:, i, :, :, :]
						GT = GT_patches[:, i, :, :, :]
						images = images.to(self.device)
						GT = GT.to(self.device)
						
						# # check images and GT
						# self.check_images(images, GT)
					
						# # check binning fuction
						# gt_btlnek = self.bin_gt(GT)
						# self.check_bin(gt_btlnek, GT, i)

						
					

						# SR : Segmentation Result
						SR, image_btlnek = self.unet(images)
						SR_probs = F.sigmoid(SR)
						SR_flat = SR_probs.view(SR_probs.size(0),-1)

						GT_flat = GT.view(GT.size(0),-1)
						loss_byend = self.diceLoss(SR_flat,GT_flat)
						#loss = self.criterion(SR_flat,GT_flat)
						#epoch_loss += loss.item()

						# find loose at bottle neck
						gt_btlnek = self.bin_gt(GT)
					
						conv = nn.Conv2d(self.BottleNeck_size, 1, kernel_size=1).to(self.device)
						image_btlnek = conv(image_btlnek.to(self.device))
						img_probs = F.sigmoid(image_btlnek)
						loss_btlnek = self.diceLoss(img_probs.view(img_probs.size(0), -1), gt_btlnek.view(gt_btlnek.size(0), -1))
					
						# calculate full loss
						loss_ratio = 0.75
						loss = (1-loss_ratio)*loss_byend + loss_ratio*loss_btlnek
						epoch_loss += (1-loss_ratio)*loss_byend.item() + loss_ratio*loss_btlnek.item()
					

						# Backprop + optimize
						self.reset_grad()
						loss.backward()
						self.optimizer.step()

						acc += get_accuracy(SR,GT)
						SE += get_sensitivity(SR,GT)
						SP += get_specificity(SR,GT)
						PC += get_precision(SR,GT)
						F1 += get_F1(SR,GT)
						JS += get_JS(SR,GT)
						DC += get_DC(SR,GT)
						length += images.size(0)
						

						# Visualization 
												
						imagesV = [images[0, 0, :, :], images[1, 0, :, :], images[2, 0, :, :]]  
						GTsV = [GT[0, 0, :, :], GT[1, 0, :, :], GT[2, 0, :, :]]
						SR_probsV = [SR_probs[0, 0, :, :], SR_probs[1, 0, :, :], SR_probs[2, 0, :, :]]
						img_probsV = [img_probs[0, 0, :, :], img_probs[1, 0, :, :], img_probs[2, 0, :, :]]
						self.visualiz_processing(imagesV, GTsV, SR_probsV, img_probsV, fig, axes)
						#plt.tight_layout()
						#plt.show()


				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length

				# Print the log info
				print("[INFO] EPOCH: {}/{}".format(epoch+1, self.num_epochs))
				print("Train loss: {:.6f}".format(epoch_loss))
				
				# print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
				# 	  epoch+1, self.num_epochs, \
				# 	  epoch_loss,\
				# 	  acc,SE,SP,PC,F1,JS,DC))

			

				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				for i, (images, GT) in enumerate(self.valid_loader):
					for i in range(images_patches.size(1)):
						images = images_patches[:, i, :, :, :]
						GT = GT_patches[:, i, :, :, :]
						images = images.to(self.device)
						GT = GT.to(self.device)
						
						images = images.to(self.device)
						GT = GT.to(self.device)
						SR, _ = self.unet(images)
						SR = F.sigmoid(SR)
						
						loss_val = self.diceLoss(SR_flat,GT_flat)
						#loss = self.criterion(SR_flat,GT_flat)
						
						val_loss += loss_val.item()

						acc += get_accuracy(SR,GT)
						# SE += get_sensitivity(SR,GT)
						# SP += get_specificity(SR,GT)
						# PC += get_precision(SR,GT)
						# F1 += get_F1(SR,GT)
						# JS += get_JS(SR,GT)
						# DC += get_DC(SR,GT)
						
						length += images.size(0)
					
				# acc = acc/length
				# SE = SE/length
				# SP = SP/length
				# PC = PC/length
				# F1 = F1/length
				# JS = JS/length
				# DC = DC/length
				# unet_score = JS + DC
				print('[Validation] loss: %.4f'%(val_loss))
				#print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
				
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''


				# Save Best U-Net model
				# if unet_score >= best_unet_score:
				# 	best_unet_score = unet_score
				# 	best_epoch = epoch
				# 	best_unet = self.unet.state_dict()
				# 	print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
				torch.save(self.unet.state_dict(),unet_path)
					
			#===================================== Test ====================================#
			del self.unet
			if best_unet is not None:
				del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT) in enumerate(self.valid_loader):
				
				for i in range(images_patches.size(1)):
						images = images_patches[:, i, :, :, :]
						GT = GT_patches[:, i, :, :, :]
						images = images.to(self.device)
						GT = GT.to(self.device)

						images = images.to(self.device)
						GT = GT.to(self.device)
						SR, _ = self.unet(images)
						SR = F.sigmoid(SR)
						acc += get_accuracy(SR,GT)
						SE += get_sensitivity(SR,GT)
						SP += get_specificity(SR,GT)
						PC += get_precision(SR,GT)
						F1 += get_F1(SR,GT)
						JS += get_JS(SR,GT)
						DC += get_DC(SR,GT)
						loss_test = self.diceLoss(SR_flat,GT_flat)
						#loss = self.criterion(SR_flat,GT_flat)
						
						test_loss += loss_test.item()
						length += images.size(0)
			print('[Test] loss: %.4f'%(test_loss))	
			endTime = time.time()
			print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
			# acc = acc/length
			# SE = SE/length
			# SP = SP/length
			# PC = PC/length
			# F1 = F1/length
			# JS = JS/length
			# DC = DC/length
			# unet_score = JS + DC


			#f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
			#wr = csv.writer(f)
			#wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
			#f.close()
			

			
