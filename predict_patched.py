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
	stride = 32
	#device = 'cpu'
	
	
	# create the output path
	output_path = path+'_mask25D_s/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
		 
	# Load the state dictionary
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	state_dict = torch.load(state_dict_path, map_location=torch.device(device))
	
	# Initialize the model
	model = U_Net(img_ch=1,output_ch=1)

	# Load the state dictionary into the model
	model.load_state_dict(state_dict)

	# Move the model to the desired device
	model = model.to(device)

	# # print model
	# for name, param in model.named_parameters():
	# 	print(name, param)
		
	# set model to evaluation mode
	model.eval()
	
	
	# get images
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
    

	# turn off gradient tracking
	with torch.no_grad():
		for index in range(len(image_paths)):
			
			# Add precceding and following images
			if (index == 0):
				image_path_prec = image_paths[index]
				image_path_foll = image_paths[index+1]
			elif (index == len(image_paths)-1):
				image_path_prec = image_paths[index-1]
				image_path_foll = image_paths[index]
			else:
				image_path_prec = image_paths[index-1]
				image_path_foll = image_paths[index+1]
			prec_image = Image.open(image_path_prec).convert("L")
			foll_image = Image.open(image_path_foll).convert("L")
				
		
			if image_paths[index].endswith('.tif'):
				
				image = Image.open(image_paths[index]).convert("L")
				
				# convert uint 16 to uint8 if needed
				# image_data = np.array(image)
				# image_uint8 = image_data.astype(np.uint8)
				# image = Image.fromarray(image_uint8)
				# image = image.convert('L')
				# predicate whole image at ones
				# img_tensor = T.ToTensor()(image).unsqueeze(0)  # Convert to tensor and add batch dimension
				# normalizer = nn.LayerNorm([1, 958, 1405])
				#normalizer = nn.LayerNorm([1, 744, 1103])
				

				# add the preceeding and following images as channels
				image = Image.merge("RGB", (prec_image, image, foll_image))# check if the image is 3 channel or not
				# predicate whole image at ones
				img_tensor = T.ToTensor()(image)
				normalizer = nn.LayerNorm([3, 958, 1405])
				img_tensor = normalizer(img_tensor)
				img_tensor = img_tensor.unsqueeze(0) 
				predMask, checkbtlnek , _, _, _=  model(img_tensor.to(device))
				output = torch.sigmoid(predMask)
				sm = nn.ReLU()
				#output = sm(predMask)



				
				

				output = output.squeeze(0)
				output = output.squeeze(0)
				output = output - output.min()
				output = output / output.max()
				#output[output >= 0.5] = 255
				output = ((output) * 255) 
				#output[output < 165] = 0
				output = output.byte()
				output_np = output.cpu().numpy()
				output_image = Image.fromarray(output_np)
				output_image.save(os.path.join(output_path,image_paths[index].split('/')[-1]))




		
		



