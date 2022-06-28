import logging

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from settings import *
from data.utils import pad


np.random.seed(RAND_SEED)


class SignalDataset(Dataset):
    """
    Consolidates the different data set so they're processed in the same way.
    Also used as the Dataset class required for PyTorch's DataLoader.

    Specifically handles:
    1. calls method to load and preprocess data for each data set
    2. pads all the signals to the same length
    3. when used as DataLoader dataset. queries the data from the data sets.
    """
    def __init__(self, corpora, args, caller='', is_adv=False, **kwargs):
        self.signal_type = args.signal_type
        assert self.signal_type in ['vel', 'pos', 'posvel', 'acc']
        self.augment_signal = args.augment
        self.input_column = 'in_{}'.format(self.signal_type)

        if caller == 'trainer':
            self.mode = 'unsupervised'
            self.split_to_val = args.use_validation_set

        else:
            self.mode = 'evaluation'
            self.split_to_val = False

        self.normalize = False
        self.corpora = corpora

        assert args.hz > 0
        self.hz = args.hz       #TODO wieso?
        self.viewing_time = args.viewing_time
        self.num_gaze_points = int(self.hz * self.viewing_time)
        self.train_set, self.val_set, self.test_set = [], [], []

        self.is_adv = is_adv
        print('Is adv training: ', self.is_adv)

        for corpus_name, corpus in self.corpora.items():
            corpus.load_data(is_adv = self.is_adv)
            corpus_samples = ['{}|{}'.format(corpus_name, i)
                              for i in range(len(corpus.data))]
              
            #print(corpus.data) # (7772, 1000, 3)

            if not args.slice_time_windows or kwargs.get('load_to_memory'):
                # here when evaluator is loaded
                signal = self._get_signal(corpus.data)[0]
                corpus.data[self.input_column] = signal
                corpus.data.drop(['x', 'y'], axis=1, inplace=True)

            if self.split_to_val:
                # here when datasets are loaded at the beginning
                train, val = train_test_split(
                    corpus_samples, test_size = 0.2, random_state=RAND_SEED)#test_size=50
            else:
                train, val = corpus_samples, []

            self.train_set.extend(train)
            self.val_set.extend(val)

        if len(self.val_set) > 0:
            self.val_set = SignalDataset_Val(self.val_set,
                                             self.corpora,
                                             self.normalize,
                                             self.signal_type)

        logging.info('\nDataset class initialized from {}.'.format(caller))
        logging.info('Hz: {}. View Time (s): {}'.format(
            args.hz, self.viewing_time))
        logging.info('Signal type: {}'.format(self.signal_type))
        logging.info('Normalize: {}'.format(self.normalize))
        logging.info('Training samples: {}'.format(len(self.train_set)))
        logging.info('Validation samples: {}'.format(len(self.val_set)))

    def _get_signal(self, df):
        # get in here when evaluator is loaded
        if self.signal_type == 'vel':
            signal = df['v']

        elif self.signal_type == 'pos':
            if self.normalize:
                signal = df[['x', 'y']].apply(self.normalize_sample, 1)
            else:
                signal = df.apply(lambda r: np.stack(r[['x', 'y']]).T, 1)

            if self.augment_signal:
                logging.info('SIGNAL AUGMENTATION NOT YET SUPPORTED')
                assert 1 == 0
        elif self.signal_type == 'acc':
            signal = df['a']

        else:
            signal = df['pv']

        if self.num_gaze_points > 0:
            signal = signal.apply(lambda x: pad(self.num_gaze_points, x))

       
        return signal, df['balancedHz']

    #fetching a data sample for a given key
    #returned values are taken as values for a called batch during the forward()
    def __getitem__(self, i):       
        corpus, idx = self.train_set[i].split('|')
        data = self.corpora[corpus].data
        #print('data[0]', data[int(idx)]) # [[2.65e-02 4.05e-02 1.00e+03] ...

        if type(data) == pd.DataFrame:
            assert False, 'data.py - if type(data) == pd.DataFrame, need to include x y vs label'
            #TODO
            #print('##################################')
            #print(data.iloc[int(idx)][self.input_column])
            #print(data.iloc[int(idx)][self.input_column][:,:-1])
            #print(data.iloc[int(idx)][self.input_column][:,-1])
            signal = data.iloc[int(idx)][self.input_column]
        # when loading data for batches: EMVIC, FIFA and ETRA
        else:  # saved time slices from corpus.py - iter_slice_chunks()
            #signal = data[int(idx)]
            # x & y values
            signal = data[int(idx)][:,:-1]  # all but last column
            # Herz Value
            labelHz = data[int(idx)][:,-1] # last column
            if self.normalize and self.signal_type != 'vel':
                signal = self.normalize_sample(signal)
       
        return signal.T, labelHz.T
        ##return signal.T, randomSR# TODO: , Herz -> astype(long) -> sind Kategorien -> direkt in CEL reingeben

    def __len__(self):
        return len(self.train_set)

    def normalize_sample(self, sample):
        if isinstance(sample, np.ndarray):
            sample[:, 0] /= MAX_X_RESOLUTION
            sample[:, 1] /= MAX_Y_RESOLUTION
            return sample

        sample.x /= MAX_X_RESOLUTION
        sample.y /= MAX_Y_RESOLUTION
        return np.array([sample.x, sample.y]).T

    def getBalancedSamplingRate(self):
        samplingRates = [30, 60, 120, 250, 500, 1000]
        chosenSamplingRate = np.random.choice(samplingRates, 1)

        return chosenSamplingRate



class SignalDataset_Val(SignalDataset):
    def __init__(self, samples, corpora, normalize, signal_type):
        self.train_set = samples
        self.corpora = corpora
        self.normalize = normalize
        self.signal_type = signal_type
