import cv2
import numpy as np
import os

# Difference of Gaussians applied to img input
def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
	img1 = cv2.GaussianBlur(img,size,sigma)
	img2 = cv2.GaussianBlur(img,size,sigma*k)
	return (img1-gamma*img2)

# Threshold the dog image, with dog(sigma,k) > 0 ? 1(255):0(0)
def edge_dog(img,sigma=0.5,k=200,gamma=0.98):
	aux = dog(img,sigma=sigma,k=k,gamma=0.98)
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] > 0):
				aux[i,j] = 255
			else:
				aux[i,j] = 0
	return aux

# garygrossi xdog version
def xdog_garygrossi(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10):
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] >= epsilon):
				aux[i,j] = 1
			else:
				ht = np.tanh(phi*(aux[i][j] - epsilon))
				aux[i][j] = 1 + ht
	return aux*255

def hatchBlend(image):
	xdogImage = xdog(image,sigma=1,k=200, gamma=0.5,epsilon=-0.5,phi=10)
	hatchTexture = cv2.imread('./imgs/hatch.jpg', cv2.IMREAD_GRAYSCALE)
	hatchTexture = cv2.resize(hatchTexture,(image.shape[1],image.shape[0]))
	alpha = 0.120
	return (1-alpha)*xdogImage + alpha*hatchTexture

# version of xdog inspired by article
def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] < epsilon):
				aux[i,j] = 1*255
			else:
				aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
	return aux


if __name__ == '__main__':
	# img = cv2.imread('imgs/rapela.jpg',cv2.IMREAD_GRAYSCALE)
	# img = cv2.resize(img,(400,400))
	# k = 1.6
	# cv2.imwrite("XDoG_Project_1.jpg", np.uint8(xdog(img, sigma=0.4, k=1.6, gamma=0.5, epsilon=-0.5, phi=10)))
	source_folder = 'DATA_PATH'
	target_folder = 'SKETCH_PATH'

	# subfolders = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_6']
	# subfolders = ['test_large', 'test_small', 'train_large', 'train_small']
	subfolders = ['v1']
	for subfolder in subfolders:
		subfolder_path = os.path.join(source_folder, subfolder)
		for file in os.listdir(subfolder_path):
			if file.endswith('.png'):
				target_dir = os.path.join(target_folder, subfolder)
				os.makedirs(target_dir, exist_ok=True)

				img_path = os.path.join(subfolder_path, file)
				img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

				k = 1.6
				
				processed_img = np.uint8(xdog(img, sigma=0.6, k=1.6, gamma=0.98, epsilon=-0.5, phi=1e20))
				
				target_path = os.path.join(target_dir, file)
				cv2.imwrite(target_path, processed_img)
				print(target_path)



	# cv2.imwrite("Original_in_Grayscale.jpg", img)
	# # 假設 edge_dog, xdog_garygrossi, xdog, hatchBlend 是已經定義好的函數
	# cv2.imwrite("Edge_DoG.jpg", edge_dog(img, sigma=0.5, k=200, gamma=0.98))
	# cv2.imwrite("XDoG_GaryGrossi.jpg", np.uint8(xdog_garygrossi(img, sigma=0.5, k=200, gamma=0.98, epsilon=0.1, phi=10)))
	# cv2.imwrite("XDoG_Project_2.jpg", np.uint8(xdog(img, sigma=1.6, k=1.6, gamma=0.5, epsilon=-1, phi=10)))
	# # Natural media (tried to follow parameters of article)
	# cv2.imwrite("XDoG_Project_3_Natural_Media.jpg", np.uint8(xdog(img, sigma=1, k=1.6, gamma=0.5, epsilon=-0.5, phi=10)))
	# cv2.imwrite("XDoG_Project_4_Hatch.jpg", np.uint8(hatchBlend(img)))
	cv2.waitKey(0)