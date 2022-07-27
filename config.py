# config the parameters for ArPLS-SF
import numpy as np

debug = True

## the input raw data size
obs_mode = 'arcdrift'
subint= int(2048)
nchan = int(65536)
npol = int(4)
beam = 19                

## parameters of the spatial filter
pol_num = 0              #the pol number to process
factor = 3               #threshold factor

## frequency range
f_min = 1000
f_max = 1500
f_samp = (f_max-f_min)/nchan
f_arr = np.arange(f_min,f_max,f_samp)

## frequency range to process
f_process = [1410,1430]
idx_chan = np.where((f_arr > f_process[0]) & (f_arr < f_process[1]))[0]
f_arr_process = f_arr[idx_chan]
nchan = int(len(f_arr))

## the frequency range to generate the threshold
f_lims = [1394,1400]
idx_chan_lims = np.where((f_arr > f_lims[0]) & (f_arr < f_lims[1]))[0]
f_arr_lims = f_arr[idx_chan_lims]

## path to save the results
path2save = '/home/wangy/spatial_filter/results/0137/1420/'


