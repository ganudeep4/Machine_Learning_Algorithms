import matplotlib.pyplot as plot
import numpy as np
import os


def compute_Z(X, centering=True, scaling=False):
    
    Z = X
    
    if centering:
        mean = np.mean(X, axis=0)
        Z = X - mean
        
    if scaling:
        std_dev = np.std(X, axis=0)
        Z = Z / std_dev
        
    return Z
        

# In[40]:


def compute_covariance_matrix(Z):
    
    COV = np.dot(np.transpose(Z), Z)
    return COV


# In[41]:


def find_pcs(COV):
    
    eig_vals, eig_vects = np.linalg.eig(COV)
    sort_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sort_indices]
    eig_vects = eig_vects[:,sort_indices]

    return eig_vals, eig_vects
    

# In[42]:


def project_data(Z, PCS, L, k, var):
            
    if (k != 0 and k <= Z.shape[1]):
        Z_star = np.dot(Z, PCS[:,:k])
        
    elif (var != 0 and var < 1):
        sum_eig_vals = np.sum(L)
        cummulative = 0
        k = 0
        for eig_value in L:
            k += 1
            cummulative = cummulative + eig_value
            cumm_percent = cummulative / sum_eig_vals
            if cumm_percent >= var:
                break
        Z_star = np.dot(Z, PCS[:,:k])
            
    return Z_star
        
