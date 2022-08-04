from scipy import io
from scipy.interpolate import interp1d
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load(filename, file_format, **kwargs):
    if file_format == 'matlab':
        return io.loadmat(filename, squeeze_me=True)
    if file_format == 'excel':
        return pd.read_excel(filename, sheet_name=kwargs['sheet'])
    if file_format == 'csv':
        return pd.read_csv(filename,
                           **kwargs)


def listdir(directory):
    # a wrapper just to prepend DATA_ROOT
    return [f for f in os.listdir(directory)
            if not f.startswith('.')]


def pad(num_gaze_points, sample):
    sample = np.array(sample)
    #-B---: Problem, sometimes len(sample) was longer than num_gaze_points(hz * viewing time), then num_zeros was negative and we couldn't pad
    num_zeros = num_gaze_points - len(sample[:num_gaze_points])
    return np.pad(sample,
                  ((0, num_zeros), (0, 0)),
                  constant_values=0)


def interpolate_nans(trial):
    nans = np.isnan(trial)
    if not nans.any():
        return trial
    nan_idxs = np.where(nans)[0]
    not_nan_idxs = np.where(~nans)[0]
    not_nan_vals = trial[not_nan_idxs]
    trial[nans] = np.interp(nan_idxs, not_nan_idxs, not_nan_vals)
    return trial


def pull_coords_to_zero(coords):
    non_neg = coords.x >= 0
    coords.x[non_neg] -= coords.x[non_neg].min()
    non_neg = coords.y >= 0
    coords.y[non_neg] -= coords.y[non_neg].min()
    return coords


def downsampleOld(trial, new_hz, old_hz):
    skip = int(old_hz / new_hz)
    trial.x = trial.x[::skip]
    trial.y = trial.y[::skip]
    return trial


def upsampleOld(trial, new_hz, old_hz):
    factor = int(new_hz / old_hz)
    num_upsampled_points = len(trial.x) * factor
    points = np.arange(0, num_upsampled_points, factor)
    new_points = np.arange(0, num_upsampled_points - (factor - 1), 1)
    trial.x = interp1d(points, trial.x.reshape(1, -1), kind='cubic'
                       )(new_points).reshape(-1)
    trial.y = interp1d(points, trial.y.reshape(1, -1), kind='cubic'
                       )(new_points).reshape(-1)
    return trial

def upsample_between_timestamp_pairs(trial, new_points, step):
    between_points = []
    for pointpair in new_points:
        start = pointpair[0]
        end = pointpair[1]
        for i in range(start+step, end, step): 
            between_points.append(i)
    
    between_points = np.array(between_points, dtype=int)
    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='linear')(between_points).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='linear')(between_points).reshape(-1)
    
    trial.timestep = np.append(trial.timestep, between_points)
    trial.x = np.append(trial.x, interpol_values_x)
    trial.y = np.append(trial.y, interpol_values_y)
    sort_indices = np.argsort(trial.timestep)
    trial.timestep = trial.timestep[sort_indices]
    trial.x = trial.x[sort_indices]
    trial.y = trial.y[sort_indices]                             
  
    return trial

def calcHz(timestep):
    if type(timestep) == pd.core.series.Series:
        timestep = timestep.to_numpy(dtype = 'float64')
    df = pd.DataFrame(timestep, columns= ['vals'])           
    temp = df[df.columns[0]]
    diff = temp.diff().mean() 
    calcHz = (1/diff) * 1000
    return calcHz

#downsample with interpol
#1000Hz -> 30Hz ->  0. 33.33333333 66.66666667 100. 133.33 ... 2000.
def downsample(trial, new_hz, old_hz):
    oldHz_calc = calcHz(trial.timestep)
    assert oldHz_calc - 10 <= old_hz <= oldHz_calc + 10
    
    step = 1000/new_hz                                     
    #FIFA & MIT 
    if type(trial['timestep']) is np.ndarray:
        max_timestep = trial['timestep'][-1]
    #ETRA, series
    elif type(trial['timestep']) is pd.core.series.Series:
        max_timestep = trial['timestep'].iloc[-1]

    new_timesteps = np.arange(0, max_timestep, step)
    
    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='linear')(new_timesteps).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='linear')(new_timesteps).reshape(-1)

    trial.timestep = new_timesteps                                                        
    trial.x = interpol_values_x                                                         
    trial.y = interpol_values_y

    newHz_calc = calcHz(trial.timestep)
    assert newHz_calc - 10 <= new_hz <= newHz_calc + 10
    
    return trial

# upsample interpol
def upsample(trial, new_hz, old_hz):
    oldHz_calc = calcHz(trial.timestep)
    assert oldHz_calc - 10 <= old_hz <= oldHz_calc + 10

    step = 1000/new_hz                                     
    #FIFA & MIT 
    if type(trial['timestep']) is np.ndarray:
        max_timestep = trial['timestep'][-1]
    #ETRA, series
    elif type(trial['timestep']) is pd.core.series.Series:
        max_timestep = trial['timestep'].iloc[-1]
    new_timesteps = np.arange(0, max_timestep, step)

    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='linear')(new_timesteps).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='linear')(new_timesteps).reshape(-1)

    trial.timestep = new_timesteps                                       
    trial.x = interpol_values_x                                                          
    trial.y = interpol_values_y  

    newHz_calc = calcHz(trial.timestep)
    assert newHz_calc - 10 <= new_hz <= newHz_calc + 10
    
    return trial


def calcPercentile(a, lowerPerc, upperPerc):
    Pmax = float('-inf')
    Pmin = float('inf')

    for i in range(a.shape[0]):
            #exclude all nans as they are getting chosen as highest value
            idx = [x for x in a[i] if ~np.isnan(x)]
            a[i] = idx
            
            maxx = np.percentile(a[i], upperPerc)
            minn = np.percentile(a[i], lowerPerc)
        
            if maxx > Pmax:
                Pmax = maxx
            if minn < Pmin:
                Pmin = minn
    return Pmin, Pmax

def getDistribution(xVals, xminP, xmaxP, yVals, yminP, ymaxP, plot):
    
    minX, maxX = calcPercentile(xVals,0,100)
    print('X: min, max: ', minX, ' ', maxX)
    minY, maxY = calcPercentile(yVals,0,100) 
    print('Y: min, max: ', minY, ' ', maxY)
    
    lowerX, upperX = calcPercentile(xVals,xminP, xmaxP)
    print('X: lower, upper Percentile: ', lowerX, ' ', upperX)
    lowerY, upperY = calcPercentile(yVals, yminP, ymaxP)
    print('Y: lower, upper Percentile: ', lowerY, ' ', upperY)
    
    if plot:
        #Plot distribution
        plt.hist(np.array(xVals), bins='auto', histtype='step')
        plt.show()
        plt.hist(np.array(yVals), bins='auto', histtype='step')
        #plt.xlim(xmin=-100, xmax = 1200)
        plt.show()



