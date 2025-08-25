
#pip install pillow

import cv2
import os
import numpy as np
import glob
from PIL import Image

calib_image_dir = "./dataset/kvasirv2_aug_5cnc/test/"

extn='/*.jpg'




calib_image_list = calib_image_dir + '/.calib_input.txt'


print("script running on folder", os.getcwd())
print("CALIB DIR", calib_image_dir)

calib_batch_size = 1

folders = glob.glob(calib_image_dir+'*')
imagenames_list = []
print(folders)
for folder in folders:
	for f in glob.glob(folder+ extn):
		imagenames_list.append(f)
#print(imagenames_list)
def calib_input(iter):
	read_images = []        
	for image in imagenames_list:
		im=Image.open(image).resize((128,128),Image.NEAREST)
	read_images.append(np.array(im)/255)
	images=np.array(read_images)
	return {"input_1":images}

def main():
	calib_input(1)

if __name__=="__main__":
	main()