# GazeMAE
This repo contains the code for my Master's thesis:
**Adversarial autoencoder for sampling rate independent gaze analysis**.

This Master's thesis is based on the work of Bautista and Naval: GazeMAE: General Representations of Eye Movements using a Micro-Macro Autoencoder, accepted to ICPR 2020. Preprint: https://arxiv.org/abs/2009.02437 (github repo: https://github.com/chipbautista/gazemae).

## Data
The data for this thesis was downloaded from:
1. **FIFA**: "Predicting human gaze using low-level saliency combined with face
    detection", Cerf M., Harel J., Einhauser W., Koch C., Neural Information Processing Systems (NIPS) 21, 2007. Get the data from [here](https://www.morancerf.com/publications). You need `general.mat`, `faces-jpg` folder, and `subjects` folder. Place them in `data/Cerf2007-FIFA` folder.
2. **ETRA**: Check ETRA 2019 Challenge [here](https://etra.acm.org/2019/challenge.html). Place the files in `data/ETRA2019/` folder.
3. **EMVIC**: Request data set from [here](http://kasprowski.pl/emvic/dataset.php). Place the files in `data/EMVIC2014/official_files/` folder.

## Folders
The following folders (red coloured) need to be created:
```diff
+├── autoencoder_bayer
-│   └── batchDisplay
-│      └── MicroMacroZ                                     (visualization of z values)
-│          └── withoutScaling
-│      └── XandY                                           (visualization of x and y values: original, destroyed, reconstructed, loss_rec)
-│          └── withoutScaling
+│   └── data                                               (datasets)
+│   └── evals                                              (classes for the evaluation of the autoencoder)
-│   └── generated-data                                     (store preprocessed data)
+│   └── network
-│   └── runs                                               (tensorboard data for loss_rec/reconstruction loss (SSE))
-│   └── runs_accuracy                                      (tensorboard data for accuarcy of the tasks)
-│   └── runs_CEL                                           (tensorboard data for loss_adv/adversarial loss (CEL))
-└── models                                                 (here are the models saved every 5 epochs, e.g. vel-e4-hz500 (velocity autoencoder trained with 500Hz, saved after epoch 4))
```

### Running the code
Open a command prompt when in "autoencoder_bayer" folder.
To run the code: 
- velocity adversarial autoencoder: python train_adversarial.py --signal-type=vel -bs=32 -hz=500
- positional adversarial autoencoder: python train_adversarial.py --signal-type=pos -bs=32 -hz=500
This trains the vel/pos model described in the Thesis, with the following settings: batch size is 32 and at 500 Hz sampling frequency. 

If you want to do the accuracy calculation just with the 80% of the features stripped from the sampling frequency information, do the following:
- velocity adversarial autoencoder: python train_adversarial.py --signal-type=vel -bs=32 -hz=500 --accuracy-calc-with-z_08

If you want the x and batch values to be visualized in the folder "batchDisplay", do the following:
- velocity adversarial autoencoder: python train_adversarial.py --signal-type=vel -bs=32 -hz=500 --plot-representations-and-batchvalues

For more adaptions have a look in the settings.py file. 

## Files/Codes
### Loading the datasets
The core Python class for loading the datasets is found in `data/corpus.py`, while the code to handle the individual datasets is in `data/corpora.py`. Each data set has a class, containing code to process the raw data.

### Preprocessing
The code for preprocessing the gaze data (normalization, cutting them up into segments) is within the SignalDataset class in data/data.py. All of the data sets are run through this class so that the preprocessing is consistent. It's also a PyTorch Dataset class, which will be later on used with PyTorch DataLoader to iterate during training.

### Autoencoder model
The architecture is defined in the files in `network/` folder. The autoencoder model is in `network/autoencoder.py`. but the actual layer functionalities are in `encoder.py` and `decoder.py`.


## Models used in this Thesis can be found in the "autoencoder_bayer/models" folder
0. AE_T
1. ...

To use the pre-trained model, you should be able to load it with `torch.load(model_file)`. For reference, all model functionalities (initializing, loading, saving) are found in `network/__init__.py`
