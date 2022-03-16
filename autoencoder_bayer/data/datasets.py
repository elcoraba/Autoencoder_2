## corpora
## just copied
from pickle import TRUE
import numpy as np
from .corpusData import EyeTrackingCorpus
from .utilsData import *

from collections import defaultdict

#handle the individual datasets 
class MIT_LowRes(EyeTrackingCorpus):
    """
    Fixations on Low-resolution Images
    Tilke Judd, Frédo Durand, Antonio Torralba
    JoV 2011

    64 subjects, 20~ images at 512px resolution
    = 1200~ trials

    Eye Tracker: ETL400 ISCAN (video-based)
    Sampling Frequency: 240Hz
    Viewing Time: 3s
    Each sample = 240 * 3 = 720 points
    average calibration error: less than 1 degree of visual angle (∼35 pixels)
    """
    def __init__(self, args):
        self.px_per_dva = 35
        self.hz = 240
        self.w, self.h = (1280, 1024)  # stimuli is 860 x 1024
        self.root = 'MIT-LOWRES/DATA/'
        self.stim_dir = self.root.replace('DATA', 'ALLSTIMULI')

        self.timestamp = []
        self.step = 4

        super(MIT_LowRes, self).__init__(args)

    def extract(self):
        data = []
        for subj in listdir(self.root):
            subj_dir = self.root + '/' + subj
            for stim in listdir(subj_dir):
                # ignore non-natural and low resolution images...
                # or should i not?
                if 'colornoise' in stim or '512.mat' not in stim:
                    continue

                mat = load(subj_dir + '/' + stim, 'matlab')
                key = list(mat.keys())[-1]
                
                self.timestamp = np.arange(0, 720*self.step, self.step) # 0, 4, 8 ... 2876
                
                # One stim has just shape (706,0), as this leads to problems with the velocity calc. This stimulus is dropped
                if mat[key]['DATA'].item()['eyeData'].item().T[0].size < 720:
                    continue

                data.append([subj,
                             stim.replace('.mat', '.jpeg'),
                             'free-viewing',  # double check
                             self.timestamp,
                             mat[key]['DATA'].item()['eyeData'].item().T[0][:720],
                             mat[key]['DATA'].item()['eyeData'].item().T[1][:720]])
                #print('shape timestep ', self.timestamp.shape)                                  #(720,)
                #print('shape x ', mat[key]['DATA'].item()['eyeData'].item().T[0][:720].shape)   #(720,)
                #For sampling frequency
                #Can't find time in data -> therefore rely on the 240Hz

        #print('MIT ', '\n', np.array(data)[0]) # ['aaa' 'i1182314083_lowRes512.jpeg' 'free-viewing' array([0, 1, ..., 720]) array([530.10126582, 530.10126582,
        return data


class ETRA2019(EyeTrackingCorpus):
    """
    8 subjects, 15 natural scenes, 2 tasks (fixation and free-viewing)
    = 240 trials
    (there are other trials.
    need to verify if i can use the 2 tasks for natural images)

    Viewing Time: 45s
    Eye Tracker: EyeLink 1000 
    Sampling Frequency: 500Hz
    """
    def __init__(self, args):
        self.w, self.h = (1024, 768) # (0, 0) at the top left, http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
        self.hz = 500
        self.px_per_dva = 35
        self.root = 'ETRA2019/data/{}'
        self.stim_dir = 'ETRA2019/images/'                                              #Stimulus images
        self.subjects = ['009', '019', '022', '058', '059',
                         '060', '062', 'SMC']

        #For sampling frequency
        self.time = defaultdict(dict)
        self.timestamp = []
        self.step = 2

        # the grey stimuli is not included in the data set, here's code to
        # generate a grey image. copy paste grey.bmp to a new 'BLANK' folder.
        # from PIL import Image
        # Image.new('RGB', (921, 630), color=(127, 127, 127)).save('grey.bmp', quality=100)

        super(ETRA2019, self).__init__(args)

    def extract(self):
        data = []
        print('In extract')

        for subj in self.subjects:                                                      # go through all 8 subjects
            subj_dir = self.root.format(subj)                                           #e.g. 'ETRA2019/data/009'                                         
            for stim in listdir(subj_dir):
                _, trial, condition, stim_type, stim_file = stim.split('_')             #stimulus name: 009_002_Fixation_Natural_nat011
                if condition == 'Fixation':                                             #TODO throw out fixations, why? Instead of 120 trials, we then just have 60 per subject
                    continue

                stim_folder = stim_type.upper()                                         # upper cases all letters in string
                if stim_type in ['Blank', 'Natural']:
                    task = stim_type + '_free-viewing'
                else:  # Waldo, Puzzle
                    task = stim_type + '_search'
                      #filename, file_format, **kwargs
                csv = load(subj_dir + '/' + stim, 'csv', delimiter=',')                 #'ETRA2019/data/009/009_002_Fixation_Natural_nat011'
                                                                                        #Time, LXpix, LYpix .... -> 326085,475.06,259.275 ...                                                                      
                #Add timestemp          
                self.timestamp = csv['Time'] - csv['Time'][0]                           #-firstTimestamp 0, 2, 4, 6...
                
                #LXpix: max: 964.66, min: -12.94, 20280    -6.22 ... 20346    -4.30
                #RXpix: max: 942.5, min: -0.0600000000000023, 20282    -0.06
                x = ((csv['LXpix'] + csv['RXpix']) / 2)                                 #XPos in pixeln #float64
                y = ((csv['LYpix'] + csv['RYpix']) / 2)
                
                # NEW: [subj, stim, task, timestep, x, y]
                data.append([
                    subj,                                                               #009
                    '/'.join([stim_folder, stim_file.replace('csv', 'bmp')]),           #NATURAL/... ?
                    task,                                                               #Natural_free_viewing
                    self.timestamp,
                    x.to_list(),                      
                    y.to_list()                       
                ])

                #For sampling frequency
                self.time[subj][trial] = csv['Time']

        return np.array(data)
        # gazeMAE: [subj, stim, task, x, y] (v is added in preprocessing)


                 

# not currently available
class EMVIC2014(EyeTrackingCorpus):
    """
    Downloaded from:
    LPiTrack: Eye Movement Pattern Recognition Algorithm and Application to
    Biometric Identification

    But official link (need to register) is here:
    Also here: http://www.kasprowski.pl/emvic/dataset.php
    Official paper:
    https://www.researchgate.net/publication/266967887_The_Second_Eye_Movement_Verification_and_Identification_Competition_EMVIC

    Two .CSV files training and test (unlabeled) used to determine the winners of the competition.
    Every line is a list of comma separated elements as follows:
    sid, known,x1, y1, x2, y2, ... xn, yn
    where:
    sid - subject identifier (sXX)
    known - decision of the subject concerning the observed image (true/false).
    xi - the i-th value of the recorded horizontal eye gaze point.
    yi - the i-th value of the recorded vertical eye gaze point.
    The values are 0 for point in the middle, positive for point on the right or lower side of the screen and
    negative for points on the left or upper side of the screen.
    Contact: Subhadeep Mukhopadhyay, Email: deep@temple.edu

    34 subjects lookeda at 20 to 50 different photographs
    = 1430 samples (837 training set, 593 test set)

    Eye Tracker: JazzNovo
    Sampling Frequency: 1000Hz
    Viewing time: 891 to 22012ms
    Task: look at face and answer if subject knows the person
    """
    def __init__(self, args):
        self.w, self.h = (None, None)
        self.hz = 1000
        self.root = 'EMVIC2014/official_files/'
        self.stim_dir = None

        super(EMVIC2014, self).__init__(args)

    def extract(self):
        data = []
        for split in ['train', 'testSolved']:
            with open(self.root + split + '.csv', 'r') as f:
                trials = [t.split(',') for t in f.read().split('\n')]

            for i, trial in enumerate(trials):
                if len(trial) == 1:
                    continue

                x = list(map(float, trial[2::2]))
                y = list(map(float, trial[3::2]))
                data.append([
                    trial[0] if split == 'train' else 'test-' + trial[0],
                    '',
                    'face',
                    # normalize so that (0, 0) is upper left of screen
                    x - np.min(x),
                    y - np.min(y)
                ])

        return np.array(data)


class Cerf2007_FIFA(EyeTrackingCorpus):
    """
    7 subjects, 250 images, but 500 trials
    Eye Tracker:  Eyelink 1000
    Sampling Frequency: 1000 Hz
    Viewing Time:  2s
    Monocular (right eye)

    7 subjects viewed a set of 250 images (1024 × 768 pixels)
    in a three phase experiment:
    1. free-viewing (200 images)
    2. search (200 images), after a probe image shown for 600ms
    3. image recognition memory (100 images)

    250 images: 200 frontal view, 50 no face but identical scene?

    https://www.morancerf.com/publications

    "Predicting human gaze using low-level saliency combined with face
    detection",
    Cerf M., Harel J., Einhauser W., Koch C.,
    Neural Information Processing Systems (NIPS) 21, 2007

    "The stimuli includes images of faces in natural scenes, including
    a set of images with anomalous faces and scenes, and an additional
    database with a variety of exposures for each image for alternating
    luminance.​
    The faces database is a set of 1024x768 pixels images which contain
    frontal faces in various sizes, locations, skin colors, races, etc.
    Each frame includes one image with no faces for comparison.alternating
    luminance.​"

    8 subjects? x 200 images
    = 1600 trials

    """
    def __init__(self, args):
        self.w, self.h = (1024, 768)

        self.root = 'Cerf2007-FIFA/subjects/'
        self.stim_dir = 'Cerf2007-FIFA/stimuli/faces-jpg/'
        self.hz = 1000
        self.px_per_dva = 36.57

        # Got this list from the accompanying general.mat
        self.stimuli = [
            ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0022', '0023', '0024', '0025', '0026', '0028', '0030', '0034', '0035', '0036', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0053', '0055', '0060', '0061', '0062', '0063', '0065', '0069', '0070', '0071', '0074', '0075', '0076', '0082', '0083', '0085', '0091', '0092', '0106', '0113', '0115', '0116', '0117', '0118', '0119', '0120', '0122', '0123', '0124', '0125', '0126', '0127', '0131', '0132', '0133', '0134', '0135', '0136', '0138', '0139', '0140', '0141', '0142', '0143', '0144', '0145', '0146', '0147', '0148', '0149', '0150', '0151', '0152', '0153', '0154', '0155', '0158', '0159', '0160', '0161', '0162', '0163', '0164', '0165', '0166', '0167', '0168', '0169', '0170', '0171', '0172', '0173', '0174', '0175', '0176', '0177', '0178', '0179', '0180', '0181', '0182', '0183', '0184', '0185', '0186', '0187', '0188', '0189', '0190', '0191', '0192', '0193', '0194', '0195', '0196', '0197', '0198', '0199', '0200', '0201', '0202', '0203', '0204', '0205', '0206', '0207', '0208', '0209', '0210', '0211', '0212', '0213', '0214', '0215', '0216', '0217', '0218', '0219', '0220', '0221', '0222', '0223', '0224', '0225', '0226', '0227', '0228', '0229', '0230', '0231', '0232', '0233', '0234', '0235', '0236', '0237', '0238', '0239', '0240', '0241', '0242', '0243', '0244', '0245', '0246', '0247', '0248', '0249', '0250', '0251', '0252', '0253', '0254', '0255', '0256', '0257', '0258', '0259', '0260', '0261', '0262', '0263', '0264', '0265', '0266', '0267', '0268', '0269', '0270'],
            ['0134', '0135', '0154', '0147', '0126', '0143', '0144', '0123', '0150', '0142', '0152', '0124', '0145', '0053', '0106', '0125', '0148', '0136', '0149', '0139', '0132', '0155', '0140', '0146', '0138', '0141', '0131', '0122', '0127', '0133', '0033', '0052', '0107', '0031', '0066', '0019', '0097', '0099', '0102', '0048', '0072', '0112', '0018', '0020', '0050', '0089', '0093', '0014', '0016', '0077', '0084', '0087', '0095', '0104', '0110', '0029', '0098', '0027', '0086', '0101', '0094', '0111', '0137', '0032', '0051', '0054', '0064', '0096', '0073', '0081', '0156', '0049', '0090', '0103', '0105', '0017', '0037', '0068', '0109', '0157', '0007', '0022', '0042', '0044', '0063', '0069', '0038', '0045', '0071', '0118', '0119', '0120', '0002', '0003', '0006', '0025', '0113', '0115', '0034', '0074', '0075', '0082', '0083', '0151', '0009', '0023', '0024', '0092', '0116', '0153', '0001', '0026', '0040', '0047', '0076', '0195', '0011', '0046', '0055', '0062', '0197', '0232', '0004', '0005', '0008', '0035', '0041', '0065', '0028', '0030', '0036', '0060', '0061', '0117', '0010', '0039', '0043', '0070', '0085', '0091', '0004', '0025', '0091', '0008', '0035', '0082', '0023', '0115', '0119', '0120', '0151', '0153', '0030', '0044', '0045', '0047', '0076', '0092', '0026', '0034', '0040', '0069', '0070', '0085', '0001', '0002', '0003', '0010', '0011', '0116', '0195', '0197', '0232', '0006', '0041', '0043', '0007', '0009', '0022', '0024', '0028', '0063', '0005', '0039', '0055', '0060', '0071', '0083', '0036', '0046', '0061', '0062', '0065', '0113', '0038', '0042', '0074', '0075', '0117', '0118'],
        ]

        #For sampling frequency
        self.time = defaultdict(dict)
        self.timestamp = []
        self.step = 1 #TODO 1000 ms / sampling rate, WIKI: frequenz

        super(Cerf2007_FIFA, self).__init__(args)

    def extract(self):
        data = []
        # Paper states 7 subjects, but has data for 8
        # (general.mat lists 34 subjects though...)
        for subj in listdir(self.root):
            if subj.startswith('_'):
                continue

            mat = load(self.root + subj, 'matlab')

            subj = mat['name']
            
            # 4 experiments in file, but only 3 has gaze data.
            # the third is supposedly the memory task, but it contains data
            # for 200 images, so idk what it's for.
            # decision: use the first 2 phases only.
            for phase, experiment in enumerate(mat['experiments'][:2]):
   
                # not the filenames but really just the order of the index
                # stimuli_order = experiment['order'].item()
                stimuli_data = experiment['scan'].item()

                task = ('free-viewing' if phase == 0
                        else ('search' if phase == 1
                              else 'memory'))

                # _tmp[subj + str(phase)] = stimuli_order
                
                for n, eye_data in enumerate(stimuli_data):
                    #Add timestemp
                    #timestamp = np.arange(0, len(eye_data['scan_x'].item()), 1)
                    #if first == True:
                    firstTimestamp = list(eye_data['scan_t'].item())[0]
                    self.timestamp = list(eye_data['scan_t'].item()) - firstTimestamp                                 # 0, 1, 2, ...
                    
                    data.append([
                        subj,
                        self.stimuli[phase][n] + '.jpg',
                        task,
                        self.timestamp,
                        list(eye_data['scan_x'].item()),
                        list(eye_data['scan_y'].item()),
                    ])
                    temp = list(eye_data['scan_t'].item())
                    self.time[subj][self.stimuli[phase][n]] = np.array(temp)
                    #TODO ist das richtig abgespeichert?
                    #TODO np.array vs nicht list

            # TO-DO: for the 2nd phase, need to filter out probe image data
        #print('FIFA ', '\n', np.array(data)[0])   
        # ['CH' '0001.jpg' 'free-viewing' array([0, 1, 2, ..., 2021]) (-> also varies) list([506.3, 506.2, 506.3, 506.3
        #print('In extract time: ' , self.time['CH']['0001'])
        return np.array(data)


class GAZEBASE(EyeTrackingCorpus):
    def __init__(self, args):
        self.px_per_dva = 35                #TODO 
        self.hz = 1000
        self.w, self.h = (1680, 1050)
        self.root = 'GazeBase'
        self.stim_dir = ''

        self.rounds = ['Round_1', 'Round_2']#, 'Round_3', 'Round_4', 'Round_5', 'Round_6', 'Round_7', 'Round_8', 'Round_9']
        self.sessions = ['S1', 'S2']
        self.abbrev = { "Balura_Game": "BLG",
                        "Fixations": "FXS",
                        "Horizontal_Saccades": "HSS",
                        "Random_Saccades": "RAN",
                        "Reading": "TEX",
                        "Video_1": "VD1",
                        "Video_2": "VD2"}

        self.step = 1

        super(GAZEBASE, self).__init__(args)

    def extract(self):
        data = []
        for rounds in self.rounds:
            print('round ', rounds)                         #Round_1
            subj_dir = self.root + '/' + rounds

            for subj in listdir(subj_dir):                  #Subject_1001
                if subj == 'Subject_1003' or subj == 'Subject_2004':
                    break
                print('subj ', subj)
                subj_idx = subj[-4:]                        #1001

                for session in self.sessions:               #S1
                    print(session)
                    session_dir = subj_dir + '/' + subj + '/' + session

                    for task in listdir(session_dir):       #S1_Balura_Game
                        csv = load(session_dir + '/' + task + '/' + f'S_{subj_idx}_{session}_{self.abbrev[task[3:]]}.csv', 'csv', delimiter=',')
                        #print(csv[:4])

                        data.append([
                            subj_idx,
                            '',                             #TODO stim?
                            task,
                            csv['n'].to_list(),
                            csv['x'].to_list(),             #Series -> List
                            csv['y'].to_list()
                        ])

                        #data = np.array(data)
        #print(np.array(data).shape)
        return np.array(data)
                

  
