## construct the spatial filter and remove the interference subspace
from sre_constants import _NamedIntConstant
import numpy as np
import config
import torch
from scipy import linalg


beam = config.beam
subint = config.subint

## calculate the covariance matrix
def make_covariance_matrix(data):
    l,m,n = data.shape
    D=np.zeros((beam,beam,m,n))
    D_filter=np.zeros((beam,beam,n))
    data_filter = data.mean(axis=1)
    for i in range(beam):
        for j in range(beam):
            D[i][j] = np.sqrt(data[i]*data[j])
            D_filter[i][j] = np.sqrt(data_filter[i]*data_filter[j])
    D_ms = np.transpose(D,(2,3,0,1)).astype('float32')
    D_filter = np.transpose(D_filter,(2,0,1)).astype('float32')
    return D_ms,D_filter       

## Eigen decomposition of the covariance matrix
def make_matrix(D_ms,D_filter,data):
    l,m,n = data.shape
    data_clean = np.zeros_like(data)
    correlation_cpu = torch.tensor(D_filter)
    u_cpu, s_cpu, v_cpu = torch.linalg.svd(correlation_cpu)
    u = np.array(u_cpu)
    spectrum = np.zeros([subint,n,beam])
    ### subject the RFI subspace ###
    rfi_components = 1
    for i in range(subint):
        for j in range(n):
            u_sample = u[int(j)].squeeze()
            u_rfi = u_sample[:,:rfi_components]
            P = np.dot(u_rfi,u_rfi.T)
            c = D_ms[i][j]
            matrix_clean = c - np.dot(P,np.dot(c,P))
            spectrum[i][j] = np.diag(matrix_clean)
    spectrum = np.transpose(spectrum,(2,0,1))
    for k in range(beam):
        data_clean[k]=spectrum[k]
    data_clean[data_clean<0]=0
    return data_clean

