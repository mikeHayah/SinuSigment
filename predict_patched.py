#import config
from re import L
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from PIL import Image, ImageOps
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T



def make_predictions(state_dict_path, path):
	
	patch_size = 256
	stride = 128
	device = 'cpu'
	
	
	# create the output path
	output_path = path+'_normalized/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
		 
	# Load the state dictionary
	state_dict = torch.load(state_dict_path, map_location=torch.device(device))
	
	# Initialize the model
	model = U_Net(img_ch=1,output_ch=1)

	# Load the state dictionary into the model
	model.load_state_dict(state_dict)

	# Move the model to the desired device
	model = model.to(device)

	# set model to evaluation mode
	model.eval()
	
	
	# get images
	paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
    

	# turn off gradient tracking
	with torch.no_grad():
		for image_path in paths:
			if image_path.endswith('.tif'):
				
				image = Image.open(image_path)
				
				# convert uint 16 to uint8 if needed
				image_data = np.array(image)
				#image_data [image_data>255] = 255	
				#image_normalized = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
				#image_normalized = (image_data - np.min(image_data)) / (1000 - np.min(image_data)) * 255
				image_uint8 = image_data.astype(np.uint8)
				
				image = Image.fromarray(image_uint8)
				image = image.convert('L')
							
				#image.save(os.path.join(output_path,image_path.split('/')[-1]))
				
				
				#image = image.crop((8, 8, image.width - 8, image.height - 8))
				img_tensor = T.ToTensor()(image).unsqueeze(0)  # Convert to tensor and add batch dimension
				#Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				#img_tensor = Norm_(img_tensor)
				img_patches = img_tensor.unfold(2, patch_size, stride).unfold(3, patch_size,stride)
				img_pro = img_patches.contiguous().view(-1, 1, patch_size, patch_size)  # Flatten patches
				
				
				predMask_patches = []
				for i in range(img_pro.size(0)):
						image = img_pro[i, :, :, :]
						image = image.unsqueeze(0)
						premask, checkbtlnek =  model(image.to(device))
						
						# # check btlneck
						# predMask = torch.sigmoid(checkbtlnek)
						
						predMask = torch.sigmoid(premask)
						predMask = predMask.squeeze(0)
						predMask_patches.append(predMask)
				
				blocks_reshaped = torch.stack(predMask_patches).squeeze(1).view(-1, patch_size * patch_size).permute(1, 0).unsqueeze(0)
				fold = nn.Fold(output_size=(img_tensor.size(2), img_tensor.size(3)), kernel_size=(patch_size, patch_size), stride=stride)
				
				# # check btlneck		
				# blocks_reshaped = torch.stack(predMask_patches)
				# conv = nn.Conv2d(1024, 1, kernel_size=1).to(device)
				# blocks_reshaped = conv(blocks_reshaped)
				# blocks_reshaped= blocks_reshaped.permute(1, 2, 3, 0)
				# blocks_reshaped = blocks_reshaped.view(1, 8 * 8, 40)	  
				# fold = nn.Fold(output_size=(40, 64), kernel_size=(8, 8), stride=8)
						
				output = fold(blocks_reshaped)
				
				ones = torch.ones_like(blocks_reshaped)
				normalization_mask = fold(ones)
				normalization_mask[normalization_mask == 0] = 1
				

				
				output = output / normalization_mask
				

				output = output.squeeze(0)
				output = output.squeeze(0)
				output = output - output.min()
				output = output / output.max()
				#output = ((output) * 255) 
				output[output >=0.5] = 255 
				output = output.byte()
				output_np = output.cpu().numpy()
				output_image = Image.fromarray(output_np)
				output_image.save(os.path.join(output_path,image_path.split('/')[-1]))




		
		



