## just copied 
from scipy import io
from scipy.interpolate import interp1d
import os
import pandas as pd
import numpy as np


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
    # Sybille: Problem, sometimes len(sample) was longer than num_gaze_points(hz * viewing time), then num_zeros was negative and we couldn't pad
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

def downsample_old(trial, new_hz, old_hz):
    skip = int(old_hz / new_hz)
    trial.x = trial.x[::skip]
    trial.y = trial.y[::skip]
    return trial

#-------------------------------------------
#downsample without interpol

#Downsample, old Hz:  1000
#Downsample, new Hz:  749.8761763249133
#1000 -> 750Hz: 0.  1.  3.  4.  5.  7.  8.  9. 11. 12. 13. 15. 16. 17. 19.
#new Hz: 299.95042141794744
#1000 -> 300Hz: 0.  3.  7. 10. 13. 17. 20. 23. 27. 30. 33. 37. 40. 43. 47.
def downsample_a(trial, new_hz, old_hz):
    #########For the comparison, type(trial['x'] -> np array
    print('In downsample WO ', trial['subj'] , ' ', trial['stim'][-10:])
    #ETRA
    #if trial['subj'] == '062' and trial['stim'] == 'NATURAL/nat014.bmp':
    #    np.savetxt(f"ETRA_500Hz_subj ID {trial['subj']}_stim {trial['stim'][-10:]}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header = str('t,x,y'))
    #FIFA
    #if trial['subj'] == 'CH' and trial['stim'] == '0001.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0002.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0042.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0055.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0001.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0038.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0113.jpg':
    #    np.savetxt(f"FIFA_1000Hz_subj ID {trial['subj']}_stim {trial['stim']}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header = 't,x,y')
    #########
    step = 1000/new_hz                                      # hier darf man nicht runden! Sonst machen wir wieder nur 1, 2, ... Schritte
    #FIFA & MIT 
    if type(trial['timestep']) is np.ndarray:
        max_timestep = trial['timestep'][-1]
    #ETRA, series
    elif type(trial['timestep']) is pd.core.series.Series:
        max_timestep = trial['timestep'].iloc[-1]
    new_timesteps = np.arange(0, max_timestep, step)
    new_timesteps = new_timesteps.round().astype(int)
    
    # If step size of original data is not 1
    # example: original hz = 200 -> original step = 5 -> Timesteps: 0, 5, 10 ...
    # if our new hz = 30 -> new step = 33,33 (rounded = 33) -> Timesteps: 0, 33, 67, 100 ...
    # As our original data just has data available in 5er steps, it does not contain a 33 timestep, therefore we need to round up resp. down: 33 -> 35, 67 -> 65 ... 
    step_oldHz = 1000/old_hz
    mod = new_timesteps % step_oldHz
    bigger_equal = np.apply_along_axis(lambda a: a >= step_oldHz/2, 0, mod)
    lower = np.apply_along_axis(lambda a: a < step_oldHz/2, 0, mod)
    # round up
    new_timesteps[bigger_equal] = new_timesteps[bigger_equal] + (step_oldHz - mod[bigger_equal]) 
    # round down
    new_timesteps[lower] = new_timesteps[lower] - mod[lower]
 
    idx = np.intersect1d(trial.timestep, new_timesteps, return_indices=True)[1]                                               
    trial.timestep = trial.timestep[idx]
    trial.x = trial.x[idx]
    trial.y = trial.y[idx]
    ########calculate new sampling freq.
    '''
    print(old_hz)
    df = pd.DataFrame(np.array(trial.timestep), columns= ['vals'])            
    temp = df[df.columns[0]]
    diff = temp.diff().mean() 
    newHz = (1/diff) * 1000
    print(newHz)
    exit()
    '''
    #########For the comparison, type(trial['x'] -> np array
    #ETRA
    #if trial['subj'] == '062' and trial['stim'] == 'WALDO/wal003.bmp' or trial['subj'] == '062' and trial['stim'] == 'NATURAL/nat014.bmp' or trial['subj'] == '022' and trial['stim'] == 'WALDO/wal004.bmp' or trial['subj'] == '022' and trial['stim'] == 'PUZZLE/puz010.bmp' or trial['subj'] == '009' and trial['stim'] == 'PUZZLE/puz013.bmp' or trial['subj'] == '009' and trial['stim'] == 'NATURAL/nat004.bmp':
    #    np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_wo_interpol_subj ID {trial['subj']}_stim {trial['stim'][-10:]}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header = str('t,x,y'))
    #    print('Next')
        #exit()
    #FIFA
    #if trial['subj'] == 'CH' and trial['stim'] == '0001.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0002.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0042.jpg' or trial['subj'] == 'CH' and trial['stim'] == '0055.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0001.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0038.jpg' or trial['subj'] == 'JV' and trial['stim'] == '0113.jpg':
    #    np.savetxt(f"FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj ID {trial['subj']}_stim {trial['stim']}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header = 't,x,y')
    ##########

   
    return trial

#downsample with interpol
#1000Hz -> 30Hz ->  0. 33.33333333 66.66666667 100. 133.33 ... 2000.
def downsample(trial, new_hz, old_hz):
    print('Downsample interpol ', trial['subj'] , ' ', trial['stim'])
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
    
    return trial

# upsample interpol
def upsample(trial, new_hz, old_hz):
    #print('In upsample'.upper())

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
    '''
    #print(old_hz)
    df = pd.DataFrame(new_timesteps, columns= ['vals'])            
    temp = df[df.columns[0]]
    diff = temp.diff().mean() 
    newHz = (1/diff) * 1000
    print(newHz)
    '''
    
    return trial

#19570375   19570382 -> new_points = ...376, ...377, ..., ...381
def upsample_between_timestamp_pairs(trial, new_hz, old_hz, new_points, step):
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
    #print('BEFORE, length trial.timestep: ', len(trial.timestep))
    trial.timestep = np.append(trial.timestep, between_points)
    trial.x = np.append(trial.x, interpol_values_x)
    trial.y = np.append(trial.y, interpol_values_y)
    #print('AFTER, length trial.timestep: ', len(trial.timestep))
    #axes: [Index(['subj', 'stim', 'task', 'timestep', 'x', 'y'], dtype='object')]
    sort_indices = np.argsort(trial.timestep)
    trial.timestep = trial.timestep[sort_indices]
    trial.x = trial.x[sort_indices]
    trial.y = trial.y[sort_indices]                             #->[170094 170095 170096 170097 170098 170099 170100 170101 170102 170103 170104 170105 170106 170107 170108 170109]
                                                                #print(trial.timestep[1640:1656])
    #print('SORTED ', trial.timestep[-20:])
    #print('is trial timestep sorted AFTER sort', np.all(np.diff(trial.timestep) >= 0))
    return trial

    '''
    Example: 
    point pair: (170096, 170108)
    timestemp           x              y
    170096          0.90968252      0.
    ----------------------------------------------interpol points
    170097          0.91118736      0.0093068
    170098          0.90722695      0.02862736
    170099          0.89879121      0.05595665
    170100          0.88687004      0.08928966
    170101          0.87245337      0.12662136
    170102          0.85653111      0.16594673
    170103          0.84009315      0.20526075
    170104          0.82412942      0.24255839
    170105          0.80962983      0.27583462
    170106          0.79758428      0.30308444
    170107          0.78898269      0.32230281
    ----------------------------------------------interpol points
    170108          0.78481497      0.33148471


    Example: CH stim:  0117.jpg
    point pair: (1610, 1667)
    timestemp           x              y
    1610          0.59143851      0.72141541
    ----------------------------------------------interpol points
    1611          0.59408523      0.72651082
    1612          0.59634838      0.73035547
    ...

    1665         0.50041198       0.44834106
    1666         0.5008119        0.45845911
    ----------------------------------------------interpol points
    1667         0.5015264        0.47006027
    '''
    




