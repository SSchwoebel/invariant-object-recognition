import numpy as np
import matplotlib.pyplot as pl
import Image
import scipy.signal as sg

#some variable initializations

#resolution of gabor filter
resolution = 1.
#size of gabor filter
gsize = 30
#Number of gabor filter orientations with cosine in the gabor bank
N_Greal = 8
#Number of gabor filter orientations with sine in the gabor bank
N_Gimag = 0
#number of different wave vectors in the gabor bank
N_Size = 8
#total number of gabor filters
N_Gabor = N_Greal*N_Size+N_Gimag*N_Size

# return 2D Gabor Filter with cosine. Uses multivariate Gaussian with standard deviations "sigmax" and "sigmay" and has a mean of 0. Cosine has wave vector "k", phase "phi and is rotated around angle "theta". Filter has "size" as size with resolution "res". 
def Gabor_real(size, sigmax, sigmay, k, phi, theta, res):
    x,y = np.mgrid[-size/2:size/2:res,-size/2:size/2:res]
    xrot = x*np.cos(theta) + y*np.sin(theta)
    return (1/(2.*np.pi*sigmax*sigmay))*np.exp(-(x**2/(2.0*sigmax**2))-(y**2/(2.0*sigmay**2)))*np.cos((k*xrot)-phi)

# return 2D Gabor Filter with sine. Uses multivariate Gaussian with standard deviations "sigmax" and "sigmay" and has a mean of 0. Sine has wave vector "k", phase "phi and is rotated around angle "theta". Filter has "size" as size with resolution "res". 
def Gabor_imag(size, sigmax, sigmay, k, phi, theta, res):
    # return 2D Gabor Filter
    x,y = np.mgrid[-size/2:size/2:res,-size/2:size/2:res]
    xrot = x*np.cos(theta) + y*np.sin(theta)
    return (1/(2.*np.pi*sigmax*sigmay))*np.exp(-(x**2/(2.0*sigmax**2))-(y**2/(2.0*sigmay**2)))*np.sin((k*xrot)-phi)

# return gabor bank of "n_real" cosine gabor filters and "n_imag" sine gabor filters with "n_size" wave vektors and size "size" and resolution "res". returns array of gabor filters with shape (N_Gabor,int(size/res),int(size/res) such that gabor_bank[i] is the i-th gabor filter. gabor_bank[0:nsize*n_real] contains the real gabor filters where gabor_bank[0:n_real] contains n_real differently sized filters of the same orientation and so on. gabor_bank[nsize*n_real:nsize*(n_real+n_imag)] contains the imaginary gabor filters.
def Gabor_Bank(n_real, n_imag, n_size, size, res):
	#total number of gabor filters in the gabor bank
	N_Gabor = n_real*n_size+n_imag*n_size
	gabor_bank = np.zeros((N_Gabor,int(size/res),int(size/res)))
	for i in range(n_real):
		for j in range(n_size):
			gabor_bank[i*n_size+j] = Gabor_real(size,j/4.+1/2.,j/4.+1/2.,n_size/2.+1-j/2.,0,i*np.pi/n_real,res)
	for i in range(n_imag):
		for j in range(n_size):
			gabor_bank[i*n_size+j+n_real*n_size] = Gabor_imag(size,j/4.+1/4.,j/4.+1/4.,n_size/2.+1-j/2.,0,i*2*np.pi/n_imag,res)
	return gabor_bank
	
#nice gabor filter plot function for the "N"-th gabor filter. for my 4 different sizes though.	
def Gabor_Plot(gabor_bank,N):
	f,ar = pl.subplots(2,2)
	ar[0,0].imshow(gabor_bank[N+0])
	ar[0,1].imshow(gabor_bank[N+1])
	ar[1,0].imshow(gabor_bank[N+2])
	ar[1,1].imshow(gabor_bank[N+3])
	f.show()

#reads png image with name "image_name". returns a 2D numpy array
def Read_Image(img_name):
	img = Image.open(img_name).convert('LA')
	img = np.array(img)
	#img = img[:,:,0]
	#img = np.dot(img[:,:,:3], [0.299, 0.587, 0.144])
	return img

#plots image after reading. also plots convolved image if given cimg[i] as argument
def Plot_Image(img):
	pl.figure()
	pl.imshow(img,cmap='gray')
	pl.show()

#convolve data
def Convolve_Data(img,gabor_bank):
	cimg = np.zeros((gabor_bank.shape[0],gabor_bank.shape[1]+img.shape[0]-1,gabor_bank.shape[2]+img.shape[1]-1))
	for i in range(gabor_bank.shape[0]):
		cimg[i]=sg.convolve2d(img, gabor_bank[i])
		#return status of convolution (since it is soo slow)
		print N_Gabor, i
	return cimg

#write "data" into "filename". checks data after writing with assertion.
def Write_Data(data,filename):
	with file(filename, 'w') as outfile:
		for i in range(data.shape[0]):
			np.savetxt(outfile, data[i])			
	new_data = np.loadtxt(filename)
	new_data = new_data.reshape((data.shape[0],data.shape[1],data.shape[2]))
	assert np.all(new_data == data)
	
def Read_Img_Database():
	for i in range(1,101):
		for j in range(356):
			filename="coil-100/obj"+str(i)+"__"+str(j)+".png"
			
img=Read_Image('coil-100/obj1__100.png')
Plot_Image(img)
