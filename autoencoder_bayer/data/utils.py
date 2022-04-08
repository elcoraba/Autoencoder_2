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
    #num_zeros = num_gaze_points - len(sample)
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


#-B-------------------------------------------------------------------------------------------------
def upsample_between_timestamp_pairs(trial, new_points, step):
    between_points = []
    # (19570375   19570382)
    for pointpair in new_points:
        start = pointpair[0]
        end = pointpair[1]
        for i in range(start+step, end, step): 
            between_points.append(i)
    
    between_points = np.array(between_points, dtype=int)
    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='linear')(between_points).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='linear')(between_points).reshape(-1)
    '''
    print('point pair ', pointpair)
    print('interpol x ', interpol_values_x)
    print('interpol y ', interpol_values_y)
    print(type(trial.x))
    print(trial.x.shape)
    print('trial x ',trial.x[trial.timestep == pointpair[0]], '...', trial.x[trial.timestep == pointpair[1]])
    print('trial y ',trial.y[trial.timestep == pointpair[0]], '...', trial.y[trial.timestep == pointpair[1]])
    '''
    # trial type = Series
    trial.timestep = np.append(trial.timestep, between_points)
    trial.x = np.append(trial.x, interpol_values_x)
    trial.y = np.append(trial.y, interpol_values_y)
    sort_indices = np.argsort(trial.timestep)
    trial.timestep = trial.timestep[sort_indices]
    trial.x = trial.x[sort_indices]
    trial.y = trial.y[sort_indices]                             #->[170094 170095 170096 170097 170098 170099 170100 170101 170102 170103 170104 170105 170106 170107 170108 170109]
                                                                #print(trial.timestep[1640:1656])
    #print('SORTED ', trial.timestep[-20:])
    #print('is trial timestep sorted AFTER sort', np.all(np.diff(trial.timestep) >= 0))
    return trial

def calcHz(timestep):
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
    
    #print('Downsample interpol ', trial['subj'] , ' ', trial['stim'])
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

    #ETRA
    #if trial['subj'] == '062' and trial['stim'] == 'WALDO/wal003.bmp' or trial['subj'] == '062' and trial['stim'] == 'NATURAL/nat014.bmp' or trial['subj'] == '022' and trial['stim'] == 'WALDO/wal004.bmp' or trial['subj'] == '022' and trial['stim'] == 'PUZZLE/puz010.bmp' or trial['subj'] == '009' and trial['stim'] == 'PUZZLE/puz013.bmp' or trial['subj'] == '009' and trial['stim'] == 'NATURAL/nat004.bmp':
    #    np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_w_interpol_linear_subj ID {trial['subj']}_stim {trial['stim'][-10:]}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header = str('t,x,y'))
    #FIFA
    #if trial['subj'] == 'CH' and trial['stim'] == '0001.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0002.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0042.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0055.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0001.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0038.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0113.jpg':
    #    np.savetxt(f"FIFA_1000Hz_to_30Hz_Downsample_w_interpol_cubic_subj ID {trial['subj']}_stim {trial['stim']}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header = 't,x,y')
    #####################

    newHz_calc = calcHz(trial.timestep)
    assert newHz_calc - 10 <= new_hz <= newHz_calc + 10
    
    return trial

# upsample interpol
def upsample(trial, new_hz, old_hz):
    oldHz_calc = calcHz(trial.timestep)
    assert oldHz_calc - 10 <= old_hz <= oldHz_calc + 10
    
    ########################################for comparison
    '''
    stim = '0113.jpg'
    subj = 'JV'
    folder1 = 'Downsample_Comparison'
    folder2 = f'FIFA_1000Hz_subj {subj}_stim {stim}'
    fileL = f'FIFA_1000Hz_to_30Hz_Downsample_w_interpol_linear_subj ID {subj}_stim {stim}.csv'
    fileC = f'FIFA_1000Hz_to_30Hz_Downsample_w_interpol_cubic_subj ID {subj}_stim {stim}.csv'
    fileO = f'FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj ID {subj}_stim {stim}.csv'
    
    x = np.loadtxt(f"{folder1}/{folder2}/{fileL}", delimiter=',', skiprows=1)
    x = np.transpose(x)
    trial.timestep = x[0]
    trial.x = x[1]
    trial.y = x[2]
    new_hz = 1000               #!WICHTIG, ETRA: 500, FIFA: 1000, MIT: 250
    '''
    ########################################

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
    # 1000Hz -> 1200Hz: 0    1    2 ... 2018 2019 2020 -> 0.00000000e+00 8.33333333e-01 1.66666667e+00 ... 2.01750000e+03 2.01833333e+03 2.01916667e+03
    # length: 2021 -> 2424
    trial.timestep = new_timesteps                                       
    trial.x = interpol_values_x                                                         #(2877,) 
    trial.y = interpol_values_y  
    
    #np.savetxt(f"FIFA_30Hz_wIntLin_to_1000Hz_Upsample_w_interpol_linear_subj ID {subj}_stim {stim}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header='t,x,y', comments = '')
    #exit()

    newHz_calc = calcHz(trial.timestep)
    assert newHz_calc - 10 <= new_hz <= newHz_calc + 10
    
    return trial

#-B-----------------

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



