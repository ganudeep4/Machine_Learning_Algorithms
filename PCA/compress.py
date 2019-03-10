import matplotlib.pyplot as plot
import numpy as np
import os
from pca import *

# 2. Application part in Project
#     

# In[53]:


def load_data (input_dir):
    
    images = os.listdir(input_dir)    
    reading_first_img = True
    count = 0
    
    for image in images:
        count += 1
        
        temp_img = plot.imread(input_dir+image)
        temp_img = temp_img.astype(float)
        temp_img = temp_img.flatten('C')
        temp_img  = np.reshape(temp_img, (1, temp_img.shape[0]))
        
        if reading_first_img:
            img_arrays = np.empty((0, temp_img.shape[0]), float)
            img_arrays = temp_img
            reading_first_img = False
        else:
            img_arrays = np.append(img_arrays, temp_img, axis=0)
    return img_arrays.transpose()
    


# In[57]:


def compress_images(DATA, k):
    Z = compute_Z(DATA)
    COV = compute_covariance_matrix(Z)
    L, PCS = find_pcs(COV)
    Z_star = np.dot(Z, PCS[:,:k])
    
    x_compress = np.dot(Z_star, PCS[:,:k].transpose())
    
    if not os.path.exists('Output'):
        os.makedirs('Output')
    
    for i in range(DATA.shape[1]):
        face = np.reshape(x_compress[:,i], (60,48))
        plot.imsave('Output/image_'+str(i+1)+'.png', face, cmap='gray')

    

