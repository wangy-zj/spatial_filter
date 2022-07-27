# -*- coding: UTF-8 -*- 
from importlib.resources import path
import time
import os,sys
from threading import Thread
import warnings
import numpy as np

import config
import spatial_filter as sf
import read_write as rw
import baseline_removal

warnings.filterwarnings('ignore')

### main code ###
if __name__=='__main__':
    time_start = time.time()
    beam = config.beam
    filepath = sys.argv[1]

    if not os.path.exists(config.path2save):
        os.mkdir(config.path2save)
        os.mkdir(config.path2save+'/raw')
        os.mkdir(config.path2save+'/baseline')
        os.mkdir(config.path2save+'/clean')
        os.mkdir(config.path2save+'/mask')

	
    ### read the original fits files
    data,data_lims,filename = rw.read_fits(filepath)

    if filename[0].find(config.obs_mode) != -1:
        source_name = filename[0][filename[0].find('/Dec'):filename[0].find('_'+config.obs_mode)][1:]
        print('Observation name:',source_name)
    else:
        sys.exit('Wrong observation mode!')

    ### baseline removal
    data_baseline,baseline = baseline_removal.baseline_removal(data)
    data_baseline_lims,baseline_lims = baseline_removal.baseline_removal(data_lims)
    del baseline_lims

    if config.debug:
        print('data_process shape is :',data_baseline.shape,', data type is:',data_baseline.dtype)

    ### spatial filter
    D_ms,D_filter =  sf.make_covariance_matrix(data_baseline)
    D_ms_lims,D_filter_lims = sf.make_covariance_matrix(data_baseline_lims)
    if config.debug:
        print('Cov_matrix shape is: ',D_ms.shape)
        print('Cov_matrix for filter is: ',D_filter.shape,';data type is:',D_filter.dtype)
    d_clean = sf.make_matrix(D_ms,D_filter,data)
    d_clean_lims = sf.make_matrix(D_ms_lims,D_filter_lims,data_lims)
    if config.debug:
        print('d_clean shape is : ', d_clean.shape,'data type is:',d_clean.dtype)

    ### flagging RFI and generating mask files
    for i in range(beam):
        rw.out(data_baseline[i],data_baseline_lims[i],d_clean[i],d_clean_lims[i],baseline[i],filename[i],source_name)

