#tensorboard --logdir=runs
import time
import logging
from datetime import datetime

import numpy as np
from torch import manual_seed, nn, device, cuda, multiprocessing, cat, tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import get_corpora
from data.data import SignalDataset
from network import ModelManager
from evaluate import *
from evals.classification_tasks import *
from evals.utils import *
from settings import *

from tqdm import tqdm

np.random.seed(RAND_SEED)
manual_seed(RAND_SEED)


class CSVAE:
    def __init__(self):
        self.model = ModelManager(args)

        self.save_model = args.save_model
        self.cuda = args.cuda
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        if self.cuda and self.device.type == 'cuda':
            self.model.network = self.model.network.cuda()
        else:
            self.model.network = self.model.network.to(self.device)

        #self.rec_loss = args.rec_loss #MSE
        self.hz = args.hz

        #-B----
        # introduced two summary writer, so we can see the two functions in one graph in tensorboard
        self.tensorboard_train = SummaryWriter(f"runs/{run_identifier}_trainLoss")
        self.tensorboard_val = SummaryWriter(f"runs/{run_identifier}_valLoss")
        self.tensorboard_first_epoch_train = SummaryWriter(f"runs/{run_identifier}_FIRSTepoch_trainLoss")
        self.tensorboard_first_epoch_val = SummaryWriter(f"runs/{run_identifier}_FIRSTepoch_valLoss")

        self.CELtensorboard_train = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_trainLoss")
        self.CELtensorboard_val = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_valLoss")
        self.CELtensorboard_first_epoch_train = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_FIRSTepoch_trainLoss")
        self.CELtensorboard_first_epoch_val = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_FIRSTepoch_valLoss")
        #-B----

        self._load_data()
        self._init_loss_fn(args)
        self._init_evaluator()

    def _load_data(self):
        self.dataset = SignalDataset(get_corpora(args), args,
                                     caller='trainer')

        _loader_params = {'batch_size': args.batch_size, 'shuffle': True,
                          'pin_memory': True}

        if len(self.dataset) % args.batch_size == 1:
            _loader_params.update({'drop_last': True})

        self.train_dataloader = DataLoader(self.dataset, **_loader_params)
        self.val_dataloader = (
            DataLoader(self.dataset.val_set, **_loader_params)
            if self.dataset.val_set else None)

    def _init_loss_fn(self, args):
        #self._loss_types = ['total']
        #self._loss_types.append('rec')
        self.loss_fn = nn.MSELoss(reduction='none')
        
        self.loss_adv = nn.CrossEntropyLoss()

    def _init_evaluator(self):
        print('################################################################################################')
        print('##################################Init evaluator################################################')
        print('################################################################################################')
        # for logging out this run
        _rep_name = '{}{}-hz{}-s{}'.format(run_identifier, 'mse', self.dataset.hz, self.dataset.signal_type)

        self.evaluator = RepresentationEvaluator(
            tasks=[Biometrics_EMVIC(), ETRAStimuli(),
                   AgeGroupBinary(), GenderBinary()],
            ## classifiers='all',
            classifiers=['svm_linear'],
            args=args, model=self.model,
            # the evaluator should initialize its own dataset if the trainer
            # is using manipulated trials (sliced, transformed, etc.)
            dataset=(self.dataset if not args.slice_time_windows
                     else None),
            representation_name=run_identifier, #_rep_name,
            # to evaluate on whole viewing time
            viewing_time=-1)

        if args.tensorboard:
            self.tensorboard_acc_train = SummaryWriter('runs_accuracy/train/{}_acc'.format(run_identifier)) #_rep_name
            self.tensorboard_acc_val = SummaryWriter('runs_accuracy/val/{}_acc'.format(run_identifier))
        else:
            self.tensorboard_acc_train = None
            self.tensorboard_acc_val = None
    
   #-B---- #was self.running_loss[dset]['total'] -> now self.running_loss[dset]
   #saves loss every 100 batches
   # was reset_running_loss_100
    def init_running_loss_100(self):
        self.running_loss_100 = {'train': 0.0,
                                 'CELtrain': 0.0}

    #adds up to the epoch loss
    def init_running_loss(self):
        self.running_loss = {'train': 0.0,'val': 0.0,
                             'CELtrain': 0.0, 'CELval': 0.0}
    
    # was init_global_losses_100
    def init_epoch_losses_100(self, num_checkpoints):
        amount_checkpoints = int(num_checkpoints * (int(len(self.train_dataloader)/TRAIN_SAVE_LOSS_EVERY_X_BATCHES)) ) +1
        self.global_losses_100 = {
            'train': np.zeros(amount_checkpoints),
            'val': np.zeros(amount_checkpoints),
            'CELtrain': np.zeros(amount_checkpoints),
            'CELval': np.zeros(amount_checkpoints)}
       
    # was init_global_losses
    def init_epoch_losses(self, num_checkpoints):
        self.epoch_losses = {
            'train': np.zeros(num_checkpoints),
            'val': np.zeros(num_checkpoints),
            'CELtrain': np.zeros(num_checkpoints),
            'CELval': np.zeros(num_checkpoints),
            }

    #just used for first epoch, saves loss for every batch
    def init_currentLoss(self):
        self.currentLoss = {'MSE': 0.0, 'CEL': 0.0}
    
   #-B----  

   #TODO Add early stopping?

   #TODO Add batch_to_color & batch_diff_to_color?

    def train(self):
        logging.info('\n===== STARTING TRAINING WITH ADVERSARIAL =====')
        logging.info('{} all samples, {} train batches, {} val batches'.format(len(self.dataset), len(self.train_dataloader), len(self.val_dataloader)))
        logging.info('Loss Functions:' + str(self.loss_fn) + ', ' + str(self.loss_adv))  #TODO Two loss funct

        self.init_epoch_losses(TRAIN_NUM_EPOCHS)
        self.init_epoch_losses_100(TRAIN_NUM_EPOCHS)
        self.init_currentLoss()
        counter_100 = 0

        t = tqdm(range(0, ADV_TRAIN_NUM_EPOCHS))
        for e in t:
        #while i < MAX_TRAIN_ITERS:
            self.init_running_loss()
            self.init_running_loss_100()

            self.model.network.train()
            for b, batch in enumerate(tqdm(self.train_dataloader, desc = 'Train Batches')):
                sample, sample_rec = self.forward(batch)

                #-B----
                #save running loss 100 and reset it
                if (b+1) % TRAIN_SAVE_LOSS_EVERY_X_BATCHES == 0:
                    mean_loss = self.running_loss_100['train'] / TRAIN_SAVE_LOSS_EVERY_X_BATCHES
                    self.global_losses_100['train'][counter_100] = mean_loss                       #e*len(self.dataloader) + b
                    self.tensorboard_train.add_scalar(f"MSE loss_per_{TRAIN_SAVE_LOSS_EVERY_X_BATCHES} batches", mean_loss, counter_100)
                    #---
                    CELmean_loss = self.running_loss_100['CELtrain'] / TRAIN_SAVE_LOSS_EVERY_X_BATCHES
                    self.global_losses_100['CELtrain'][counter_100] = CELmean_loss                       
                    self.CELtensorboard_train.add_scalar(f"CEL loss_per_{TRAIN_SAVE_LOSS_EVERY_X_BATCHES} batches", CELmean_loss, counter_100)
                    #---
                    self.init_running_loss_100()
                    counter_100 +=1
                if e < 1:
                    self.tensorboard_first_epoch_train.add_scalar(f"loss in first epoch TRAIN", self.currentLoss['MSE'], b)
                    self.CELtensorboard_first_epoch_train.add_scalar(f"CEL loss in first epoch TRAIN", self.currentLoss['CEL'], b)

                if b == 0:
                    break
                
                #-B----

            # save the train loss of the whole epoch
            self.logB(e, 'train')

            #############################################################################
            # Validate the model
            self.model.network.eval()
            for b, batch in enumerate(tqdm(self.val_dataloader, desc = 'Val Batches')):
                # In forward NN also calcs & saves the loss 
                sample_v, sample_rec_v = self.forward(batch)

                if e < 1:
                    self.tensorboard_first_epoch_val.add_scalar(f"loss in first epoch VAL", self.currentLoss['MSE'], b)
                    self.CELtensorboard_first_epoch_val.add_scalar(f"CEL loss in first epoch VAL", self.currentLoss['CEL'], b)
                
                if b == 0:
                    break
                
            #evaluation
            if (e + 1) % 1 == 0: #TODO 10
                print('---------------------Evaluation---------------------')
                #@thomas classifier bekommt nur w?
                self.evaluate_representation(sample, sample_rec, e, self.tensorboard_acc_train)
                self.evaluate_representation(sample_v, sample_rec_v, e, self.tensorboard_acc_val)

                
            self.logB(e, 'val') 
            t.set_postfix(loss = (self.epoch_losses['train'][e], self.epoch_losses['val'][e])) # print losses in tqdm bar
            # Save Model every epoch
            if self.save_model: #and (e+1)%5 == 0:
                self.model.save(e, run_identifier, self.epoch_losses, self.global_losses_100, run_identifier, args, True)
            

    def forward(self, batch):
        batch = batch.float()
        if self.cuda == True and self.device.type == 'cuda':     #batch = batch.to(self.device)
            batch = batch.cuda()

        _is_training = self.model.network.training
        dset = 'train' if self.model.network.training else 'val'
        
        #############################################################
        #decoder: out | encoder: z, mean, logvar  -> output of whole Autoencoder

        #ENCODING
        z_all, mean, logvar = self.model.network.encode(batch, cat_output=False) 
        # z_all is a list, consists of two tensors z2,z1
        # bs 64: z_all: z_0 torch.Size([64, 64]) z_1 torch.Size([64, 64])       bs 128: torch.Size([128, 64])   torch.Size([128, 64])
        # z_all[0]: bs x features #z_all[0].shape
 
        # divide batch in 80 bzw 20%
        size_w = int(0.2*len(z_all[0])) + 1      #26    
        size_z = int(0.8*len(z_all[0]))          #102   
        z2_w, z2_z = torch.split(z_all[0], [size_w, size_z])    #z2 w torch.Size([26, 64])  z torch.Size([102, 64])
        z1_w, z1_z = torch.split(z_all[1], [size_w, size_z])    #z1 w torch.Size([26, 64])  z torch.Size([102, 64])
        w = cat([z2_w, z1_w], 0)                                # torch.Size([52, 64])
        z = cat([z2_z, z1_z], 0)                                # torch.Size([204, 64])
        #print('w z ', w.shape, ' ', z.shape)                    
        #print('z ', z)
        
        #DECODING
        out_decX = self.model.network.decoder(z_all, batch, is_training=_is_training) #gets w & z
        out_adversary = self.model.network.adversary_decoder(z) 
        # TODO z.detach()? loss backward fixen?
        # TODO eventuell Netz zweimal laufen lassen, damit z_all nicht in beiden losses/backward drin ist. Erst ertes Netz traineren und auf 0 setzen, dann zweotes Netz
        #adversary needs another network! -> See TCNAUTOENCODER
        
        #loss = MSE(output, target)
        reconstructed_batch = out_decX
        loss_decX = self.loss_fn(reconstructed_batch, batch).reshape(reconstructed_batch.shape[0], -1).sum(-1).mean()
        # darf nur encoder und decoder_x anpassen dürfen, nicht adversary
        self.running_loss[dset] += loss_decX.item()
        self.currentLoss['MSE'] = loss_decX

        #TODO
        #loss = CEL, label - output [self.hz]
        #tensor_labels for 500Hz: 4 (index) (0 0 0 0 1 0 (classes: 30 60 120 250 500 1000))
        tensor_labels = torch.tensor(self.getLabel(self.hz), dtype = int)
        tensor_labels = tensor_labels.repeat(out_adversary.shape[0])        #bs 128: torch.Size([204]) # tensor([4, 4, 4, 4, ....
        loss_adv = self.loss_adv(out_adversary, tensor_labels) # input w?
        self.running_loss[str('CEL' + dset)] += loss_adv.item()
        self.currentLoss['CEL'] = loss_adv
        
        # adversary darf nicht parameter vom encoder & decoder_x ändern können nur vom adversary
        # tasks: sampling rate oder TODO subject
        # get_item: soll auch label mit rausgeben

        #update network if we are training
        if self.model.network.training:
            self.running_loss_100[dset] += loss_decX.item()
            self.running_loss_100[str('CEL' + dset)] += loss_adv.item()

            loss_decX.backward(retain_graph = True)
            
            loss_adv.backward()
            #@ Thomas  Wirken sich beide Losse auf encoder aus trotz init.py - params basis/adversarial
            
            self.model.optim_basis.step()       #decX output
            self.model.optim_basis.zero_grad()

            self.model.optim_adversarial.step()
            self.model.optim_adversarial.zero_grad()

        rand_idx = np.random.randint(0, batch.shape[0])
        return batch[rand_idx].cpu(), reconstructed_batch[rand_idx].cpu()

        ###############################################################################################################    

    def evaluate_representation(self, sample, sample_rec, i, tensorboard_acc):
        if sample is not None:
            viz = visualize_reconstruction(
                sample, sample_rec,
                filename='{}-{}'.format(run_identifier, i),
                #loss_func=self.rec_loss,
                title= 'visualize reconstruction', #'[{}] [i={}] vl={:.2f} vrl={:.2f}'.format(
                    #self.rec_loss, i, self.epoch_losses['val']['total'],
                    #self.epoch_losses['val']['rec']),
                savefig=False if tensorboard_acc else True,
                folder_name = 'train' if tensorboard_acc == self.tensorboard_acc_train else 'val'
                )

            if tensorboard_acc:
                tensorboard_acc.add_figure('e_{}'.format(i),
                                            figure=viz,
                                            global_step=i)

        self.evaluator.extract_representations(i, log_stats=False) #was true
        scores = self.evaluator.evaluate(i)
        if tensorboard_acc:
            for task, classifiers in scores.items():
                for classifier, acc in classifiers.items():
                    tensorboard_acc.add_scalar(
                        '{}_{}_acc'.format(task, classifier), acc, i)

    
    
    def logB(self, e, dset):
        def get_mean_losses(loss):
            try:
                iters = (len(self.train_dataloader) if dset == 'train'     
                         else len(self.val_dataloader))
            except TypeError:
                iters = 1
            
            if loss == 'MSE': #dset = train or val
                return (self.running_loss[dset] / iters)
            elif loss == 'CEL': #dset = CELtrain or CELval
                return (self.running_loss[str('CEL' + dset)] / iters)
        
        if dset == 'train':
            tr_loss = get_mean_losses('MSE')
            # save the mean loss of this epoch during training
            self.epoch_losses[dset][e] = tr_loss
            # reset running loss for the epoch
            self.running_loss[dset] = 0.0
            if self.tensorboard_train:
                #to_tensorboard('train', tr_loss)
                self.tensorboard_train.add_scalar(f'loss_per_epoch', tr_loss, e)
        elif dset == 'val':
            val_losses = get_mean_losses('MSE')
            self.epoch_losses[dset][e] = val_losses
            self.running_loss[dset] = 0.0
            if self.tensorboard_val:
                self.tensorboard_val.add_scalar(tag = f'loss_per_epoch', scalar_value = val_losses, global_step = e)
        #-CEL
        if dset == 'train':
            tr_loss = get_mean_losses('CEL')
            # save the mean loss of this epoch during training
            self.epoch_losses[str('CEL' + dset)][e] = tr_loss
            # reset running loss for the epoch
            self.running_loss[str('CEL' + dset)] = 0.0
            if self.CELtensorboard_train:
                self.CELtensorboard_train.add_scalar(f'CEL loss_per_epoch', tr_loss, e)
        elif dset == 'val':
            val_losses = get_mean_losses('CEL')
            self.epoch_losses[str('CEL' + dset)][e] = val_losses
            self.running_loss[str('CEL' + dset)] = 0.0
            if self.CELtensorboard_val:
                self.CELtensorboard_val.add_scalar(tag = f'CEL loss_per_epoch', scalar_value = val_losses, global_step = e)
        #-
    
    
    #Returns the index of the Label
    def getLabel(self, Hz):
        if Hz == 30:
            return 0 #[1, 0, 0, 0, 0, 0]
        elif Hz == 60:
            return 1 #[0, 1, 0, 0, 0, 0]
        elif Hz == 120:
            return 2 #[0, 0, 1, 0, 0, 0]
        elif Hz == 250:
            return 3 #[0, 0, 0, 1, 0, 0]
        elif Hz == 500:
            return 4 #[0, 0, 0, 0, 1, 0]
        elif Hz == 1000:
            return 5 #[0, 0, 0, 0, 0, 1]


args = get_parser().parse_args()
run_identifier = f"TEST-ADV-SyBa_{args.hz}Hz_{args.signal_type}_ETRA_FIFA_EMVIC" #datetime.now().strftime('%m%d-%H%M')
setup_logging(args, run_identifier)
print_settings()

logging.info('\nRUN ADVERSARIAL: ' + run_identifier + '\n')
logging.info('Arguments ', str(args))

adversarial = CSVAE()
adversarial.train()
