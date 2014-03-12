import numpy as np
import matplotlib.pyplot as pl
import Image
import scipy.signal as sg

gsize = 30
resolution = 1.
sigma_x = 5.
sigma_y = sigma_x

def Gaussian(size, res, sigmax, sigmay, mu):
	x,y = np.mgrid[-size/2:size/2:res,-size/2:size/2:res]
	return np.exp(-((x-mu[0])**2/(2.0*sigmax**2))-((y-mu[1])**2/(2.0*sigmay**2))) #(1/(2.*np.pi*sigmax*sigmay))
	
def Write_Data(data,filename):
	with file(filename, 'w') as outfile:
		for i in range(data.shape[0]):
			np.savetxt(outfile, data[i])			
	new_data = np.loadtxt(filename)
	new_data = new_data.reshape((data.shape[0],data.shape[1],data.shape[2]))
	assert np.all(new_data == data)
	
def Plot_Image(img):
	pl.figure()
	pl.imshow(img,cmap='gray')
	pl.show()
	
for i in range(20):
	for j in range(20):
		m = [float(i)-10.,float(j)-10.]
		g = Gaussian(gsize, resolution, sigma_x, sigma_y, m)
		#Plot_Image(g)
		np.savetxt('blobs/blob['+str(5+i)+','+str(5+j)+'].txt',g)
