import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import os,sys,glob
from threading import Lock, Thread
from decimal import Decimal
from copy import copy
import threading

import config

lock = threading.Lock()
beam = config.beam
f_arr_process = config.f_arr_process
idx_chan_process = config.idx_chan
f_arr_lims = config.f_arr_lims
idx_chan_lims = config.idx_chan_lims


### calculate the std for threshold 
def do_rms(data):
    matrix = data.reshape(-1)
    std = np.std(matrix,ddof=1)
    return std

### generate the mask data
def find_mask(residual,residual_lims):
    l,m = residual.shape
    flags = np.ones_like(residual)
    residual_lims = residual_lims.reshape(-1)
    threshold = np.mean(residual_lims)+ config.factor*np.std(residual_lims, ddof = 1)    
    flags[np.where(residual>threshold)] = 0
    # flag the whole time interval if half of the time samples are flagged
    for i in range(l):
        if flags[i,:].sum()<l*0.6:
            flags[i,:]= 0
    # flag the whole frequency channel if half of the frequency channels are flagged
    for j in range(m):
        if flags[:,j].sum()<m*0.6:
            flags[:,j] = 0

    return flags.astype('bool')

### plot the raw fits data
def plot_raw(data,filename,source_name):
    l,m=data.shape
    f_xticks = f_arr_process.astype('int')
    fig=plt.figure()
    basename = filename[filename.find(source_name):filename.find('.fits')]
    plt.imshow(data,
            aspect='auto',
            rasterized=True,
            interpolation='nearest',
            cmap='hot',extent=(0,m,0,l),
        )
    figure_name = config.path2save + 'raw/' + basename + '_raw' + '.png'
    plt.xlabel("Frequency(MHz)")
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    plt.ylabel("Time")
    plt.title(basename)
    plt.colorbar()
    plt.savefig(figure_name,dpi=400)
    plt.close()

    plt.plot(data.mean(axis=0),linewidth=0.7)
    plt.xlabel('Frequency(MHz)')
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    figure_name_spec = config.path2save + 'raw/' + basename + '_raw_avg.png'
    plt.savefig(figure_name_spec,dpi=400)
    plt.close()

### plot the baseline removal data
def plot_baseline(data,filename,source_name):
    l,m=data.shape
    f_xticks = f_arr_process.astype('int')
    fig=plt.figure()
    basename = filename[filename.find(source_name):filename.find('.fits')]
    plt.imshow(data,
            aspect='auto',
            rasterized=True,
            interpolation='nearest',
            cmap='hot',extent=(0,m,0,l),
        )
    figure_name = config.path2save + 'baseline/' + basename + '_baseline_removal' + '.png'
    plt.xlabel("Frequency(MHz)")
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    plt.ylabel("Time")
    plt.title(basename)
    plt.colorbar()
    plt.savefig(figure_name,dpi=400)
    plt.close()

    plt.plot(data.mean(axis=0),linewidth=0.7)
    plt.xlabel('Frequency(MHz)')
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    figure_name_spec = config.path2save + 'baseline/' + basename + '_baseline_removal_avg.png'
    plt.savefig(figure_name_spec,dpi=400)
    plt.close()

### plot the data after spatial filter
def plot_clean(data,filename,source_name):
    l,m=data.shape
    f_xticks = f_arr_process.astype('int')
    fig=plt.figure()
    basename = filename[filename.find(source_name):filename.find('.fits')]
    plt.imshow(data,
            aspect='auto',
            rasterized=True,
            interpolation='nearest',
            cmap='hot',extent=(0,m,0,l),
        )
    figure_name = config.path2save + 'clean/' + basename + '_clean' + '.png'
    plt.xlabel("Frequency(MHz)")
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    plt.ylabel("Time")
    plt.title(basename)
    plt.colorbar()
    plt.savefig(figure_name,dpi=400)
    plt.close()
    
    #plt.plot(f_arr.squeeze(),data.mean(axis=0),linewidth=0.7)
    plt.plot(data.mean(axis=0),linewidth=0.7)
    plt.xlabel('Frequency(MHz)')
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    figure_name_spec = config.path2save + 'clean/' + basename + '_clean_spec_avg.png'
    plt.savefig(figure_name_spec,dpi=400)
    plt.close()


### plot the generated mask data
def plot_mask(data,filename,source_name):
    l,m = data.shape
    palette=copy(plt.cm.hot)
    palette.set_bad('cyan', 1.0)
    basename = filename[filename.find(source_name):filename.find('.fits')]
    plt.imshow(data,
            aspect='auto',
            rasterized=True,
            interpolation='nearest',
            cmap='hot',extent=(0,m,0,l),
        )
    figure_name = config.path2save + 'mask/' + basename + '_mask' + '.png'
    plt.colorbar()
    f_xticks = f_arr_process.astype('int')
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    plt.xlabel("Frequency(MHz)")
    plt.ylabel("Subint Number")
    plt.title(basename)
    plt.savefig(figure_name,dpi=400)
    plt.close()

    plt.plot(data.mean(axis=0),linewidth=0.7)
    #plt.plot(f_arr.squeeze(),data.mean(axis=0),'r.')
    plt.xlabel('Frequency(MHz)')
    plt.xticks(np.arange(0,len(f_arr_process),len(f_arr_process)//5),f_xticks[::len(f_arr_process)//5])
    figure_name_spec = config.path2save + 'mask/' + basename + '_mask_spec_avg.png'
    plt.savefig(figure_name_spec,dpi=400)
    plt.close()

def write_data_clean(filename,d_clean,source_name):
    basename = filename[filename.find(source_name):filename.find('.fits')]
    np.savez('%s%s_clean.npz'%(config.path2save+'clean/',basename),data=d_clean.mean(axis=0))

### write the data to the .mask file as PRESTO(rfifind) 
def write_data_mask(filename,mask,source_name):
    basename = filename[filename.find(source_name):filename.find('.fits')]
    np.savez('%s%s_rfi.npz'%(config.path2save+'mask/',basename),rfi_flag=mask)

def write_data_baseline(filename,data,source_name):
    basename = filename[filename.find(source_name):filename.find('.fits')]
    np.savez('%s%s_baseline.npz'%(config.path2save+'baseline/',basename),data=data.mean(axis=0)

def write_data_raw(filename,data,source_name):
    basename = filename[filename.find(source_name):filename.find('.fits')]
    np.savez('%s%s_raw.npz'%(config.path2save+'raw/',basename),data=data.mean(axis=0)

### write the mask file and plot the figure (multiprocess)
def out(data_baseline,data_baseline_lims,d_clean,d_clean_lims,baseline,filename,source_name):
    residual = data_baseline - d_clean
    residual_lims = data_baseline_lims - d_clean_lims
    mask = find_mask(residual,residual_lims)
    data = read_fit(filename)
    data_mask = np.ma.array(data_baseline,mask=1-mask)
    
    
    plot_raw(data,filename,source_name)
    plot_baseline(data_baseline,filename,source_name)
    plot_clean(d_clean,filename,source_name)
    plot_mask(data_mask,filename,source_name)
    #write_data_raw(filename,data,source_name)
    #write_data_baseline(filename,data_baseline,source_name)
    #write_data_clean(filename,d_clean,source_name)
    #write_data_mask(filename,mask,source_name)


### read the fits files
def read_fit(filename,type='process'):
    if os.path.splitext(filename)[-1]=='.fits':
        hdulist = pyfits.open(filename)
        hdu1 = hdulist[1]
        data1 = hdu1.data['data']
        if type=='process':
            data_process = data1[:,idx_chan_process,int(config.pol_num)].squeeze()
            return data_process
        elif type=='lims':
            data_lims = data1[:,idx_chan_lims,int(config.pol_num)].squeeze()
            return data_lims

### read the 19-beam fits files one by one
def read_fits(path):
    fileList = sorted(glob.glob(path+'*.fits')) #read 19 beam fits data into data(19,T_sample,F_sample)
    data=[]
    data_lims=[]
    f_name=[]
    for i in range(beam):
        data.append(read_fit(fileList[i]))
        data_lims.append(read_fit(fileList[i],type='lims'))
        f_name.append(fileList[i])
    data=np.array(data)
    data_lims=np.array(data_lims)
    return data,data_lims,f_name
    
