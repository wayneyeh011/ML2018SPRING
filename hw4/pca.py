import numpy as np
from skimage import io
import csv
import sys
from time import time



## 0. NOTES
#  argv[1] --- dir
#  argv[2] --- image for reconstruction
#  output: reconstruction.jpg
#  keep 4 eigen face 


t00 = time()
## 1. To low dim by SVD (4 eigen face)
img_set = np.zeros([415, 1080000])
for i in range(415):
    img_set[i,:] = io.imread(sys.argv[1] + '/' + str(i) + '.jpg').flatten()   # sys.argv[1]  '../input/Aberdeen/' + str(i) + '.jpg'

img_set = img_set.astype(np.uint8)
print ('data type', img_set.dtype)




## 2. Extract eigen vectors (face)
t0 = time()
keep_dim = 4
mat_V = np.linalg.svd(img_set - img_set.mean(axis=0), full_matrices=False)[2][:keep_dim,:]
print ('SVD done, time consumption:', time()-t0)



## 3. Target data and Reconstruct
my_img = io.imread(sys.argv[1] + '/' + sys.argv[2]).reshape([1,1080000])   # '../input/Aberdeen/' + str(0) + '.jpg' 
my_img_demean = my_img - img_set.mean(axis=0)
my_img_z = my_img_demean.dot(mat_V.T)
print('low-dimension rep shape:',my_img_z.shape)
recon = ((mat_V.T).dot(my_img_z.T)).T + img_set.mean(axis=0)
print ('Reconstruct shape:',recon.shape)


recon = (recon - recon.min())
recon = recon / recon.max()
recon = (recon * 255).astype(np.uint8)
recon = recon.reshape(600,600,3)

#io.imsave('reconstruction.jpg', recon)  # reconstruction.jpg  '../output/reconstruction.jpg'  As jpg or png???
io.imsave('reconstruction.png', recon)  # reconstruction.jpg  '../output/reconstruction.jpg'  As jpg or png???
print ('Overall time consumption:', time()-t00)



## 4. Other notes
#  [AVG FACE]  avg_face = img_set.mean(axis=0).reshape([600,600,3])
#  [SVD weights]  np.linalg.svd(img_set - img_set.mean(axis=0), full_matrices=False)[1]


