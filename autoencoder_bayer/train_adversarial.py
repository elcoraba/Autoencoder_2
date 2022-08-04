#tensorboard --logdir=runs

import time
import logging
from datetime import datetime
from matplotlib.pyplot import imshow, viridis

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
import sys
from mpl_toolkits.axes_grid1 import ImageGrid

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

        self.hz = args.hz
        self.signal_type = args.signal_type
        self.acc_z_0_8 = args.accuracy_calc_with_z_08
        self.plot = args.plot_representations_and_batchvalues
        
        # introduced two summary writer, so we can see the two functions in one graph in tensorboard
        self.tensorboard_train = SummaryWriter(f"runs/{run_identifier}_trainLoss")
        self.tensorboard_val = SummaryWriter(f"runs/{run_identifier}_valLoss")
        self.tensorboard_first_epoch_train = SummaryWriter(f"runs/{run_identifier}_FIRSTepoch_trainLoss")
        self.tensorboard_first_epoch_val = SummaryWriter(f"runs/{run_identifier}_FIRSTepoch_valLoss")

        self.CELtensorboard_train = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_trainLoss")
        self.CELtensorboard_val = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_valLoss")
        self.CELtensorboard_first_epoch_train = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_FIRSTepoch_trainLoss")
        self.CELtensorboard_first_epoch_val = SummaryWriter(f"runs_CEL/CEL_{run_identifier}_FIRSTepoch_valLoss")

        self._load_data()
        self._init_loss_fn(args)
        self._init_evaluator()

    def _load_data(self):
        self.dataset = SignalDataset(get_corpora(args), args,
                                     caller='trainer', is_adv=True)

        _loader_params = {'batch_size': args.batch_size, 'shuffle': True,
                          'pin_memory': True}

        if len(self.dataset) % args.batch_size == 1:
            _loader_params.update({'drop_last': True})
        
        self.train_dataloader = DataLoader(self.dataset, **_loader_params)
        # problem with multiple values for drop_last
        if self.signal_type == 'pos' and self.hz == 1000:
            self.val_dataloader = (
                DataLoader(self.dataset.val_set, **_loader_params)
                if self.dataset.val_set else None)
        else:
            self.val_dataloader = (
                DataLoader(self.dataset.val_set, drop_last=True, **_loader_params)
                if self.dataset.val_set else None)

    def _init_loss_fn(self, args):
        self.loss_fn = nn.MSELoss(reduction='none')
        self.loss_adv = nn.CrossEntropyLoss()

    def _init_evaluator(self):
        print('################################################################################################')
        print('##################################Init evaluator################################################')
        print('################################################################################################')
        # for logging out this run
        _rep_name = '{}{}-hz{}-s{}'.format(run_identifier, 'mse', self.dataset.hz, self.dataset.signal_type)
        print('Evaluation just with 80percent of the z values: ', self.acc_z_0_8)
            
        self.evaluator = RepresentationEvaluator(
            tasks=[Biometrics_EMVIC(), ETRAStimuli(),
                   AgeGroupBinary(), GenderBinary()],
            ## classifiers='all',
            classifiers=['svm_linear'],
            is_adv = True,
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
    
   #saves loss every X batches
    def init_running_loss_100(self):
        self.running_loss_100 = {'train': 0.0,
                                 'CELtrain': 0.0}

    #adds up to the epoch loss
    def init_running_loss(self):
        self.running_loss = {'train': 0.0,'val': 0.0,
                             'CELtrain': 0.0, 'CELval': 0.0}
    
    def init_epoch_losses_100(self, num_checkpoints):
        amount_checkpoints = int(num_checkpoints * (int(len(self.train_dataloader)/TRAIN_SAVE_LOSS_EVERY_X_BATCHES)) ) +1
        self.global_losses_100 = {
            'train': np.zeros(amount_checkpoints),
            'val': np.zeros(amount_checkpoints),
            'CELtrain': np.zeros(amount_checkpoints),
            'CELval': np.zeros(amount_checkpoints)}
       
    def init_epoch_losses(self, num_checkpoints):
        self.epoch_losses = {
            'train': np.zeros(num_checkpoints),
            'val': np.zeros(num_checkpoints),
            'CELtrain': np.zeros(num_checkpoints),
            'CELval': np.zeros(num_checkpoints),
            }

    def init_currentLoss(self):
        self.currentLoss = {'MSE': 0.0, 'CEL': 0.0}

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
            self.init_running_loss()
            self.init_running_loss_100()

            self.model.network.train()
            for b, batch in enumerate(tqdm(self.train_dataloader, desc = 'Train Batches')):
                # two forward passes for the two optimizers
                sample, sample_rec = self.forward(batch, e, b) 
                self.forward_adv(batch) 

                #save running loss 100 and reset it
                if (b+1) % TRAIN_SAVE_LOSS_EVERY_X_BATCHES == 0:
                    mean_loss = self.running_loss_100['train'] / TRAIN_SAVE_LOSS_EVERY_X_BATCHES
                    self.global_losses_100['train'][counter_100] = mean_loss                       
                    self.tensorboard_train.add_scalar(f"loss_per_{TRAIN_SAVE_LOSS_EVERY_X_BATCHES} batches", mean_loss, counter_100)
                    
                    CELmean_loss = self.running_loss_100['CELtrain'] / TRAIN_SAVE_LOSS_EVERY_X_BATCHES
                    self.global_losses_100['CELtrain'][counter_100] = CELmean_loss                       
                    self.CELtensorboard_train.add_scalar(f"CEL loss_per_{TRAIN_SAVE_LOSS_EVERY_X_BATCHES} batches", CELmean_loss, counter_100)
                    
                    self.init_running_loss_100()
                    counter_100 +=1

                #draw every epoch
                if b == 0 and self.plot:
                    #batch Display
                    self.batch_to_color(sample, f"XandY", f"train-epoch{e}-original-batch")
                    self.batch_to_color(sample_rec.detach(), f"XandY", f"train-epoch{e}-reconstructed-batch")
                    self.batch_diff_to_color(sample, sample_rec.detach(), f"XandY", f"train-epoch{e}")
                
                if e < 1:
                    self.tensorboard_first_epoch_train.add_scalar(f"loss in first epoch TRAIN", self.currentLoss['MSE'], b)
                    self.CELtensorboard_first_epoch_train.add_scalar(f"CEL loss in first epoch TRAIN", self.currentLoss['CEL'], b)
                '''
                if b == 0:
                    break
                '''
                
            # save the train loss of the whole epoch
            self.logB(e, 'train')
            
            #############################################################################
            # Validate the model
            self.model.network.eval()
            for b, batch in enumerate(tqdm(self.val_dataloader, desc = 'Val Batches')):
                # In forward NN also calcs & saves the loss 
                sample_v, sample_rec_v = self.forward(batch, e, b)
                # don't need second forward here
                
                #batch Display
                if (b == (len(self.val_dataloader)-1) or b == 0) and self.plot:
                    self.batch_to_color(sample_v, f"XandY", f"val-epoch{e}-batch{b}-original-batch")
                    self.batch_to_color(sample_rec_v.detach(), f"XandY", f"val-epoch{e}-batch{b}-reconstructed-batch")
                    self.batch_diff_to_color(sample_v, sample_rec_v.detach(), f"XandY", f"val-epoch{e}-batch{b}")

                if e < 1:
                    self.tensorboard_first_epoch_val.add_scalar(f"loss in first epoch VAL", self.currentLoss['MSE'], b)
                    self.CELtensorboard_first_epoch_val.add_scalar(f"CEL loss in first epoch VAL", self.currentLoss['CEL'], b)
                '''
                if b == 0:
                    break
                '''
            
            #evaluation
            if (e + 1) % 1 == 0: #TODO 10
                print('---------------------Evaluation---------------------')
                self.evaluate_representation(sample, sample_rec, e, self.tensorboard_acc_train, True)
                self.evaluate_representation(sample_v, sample_rec_v, e, self.tensorboard_acc_val, False)

                
            self.logB(e, 'val') 
            t.set_postfix(loss = (self.epoch_losses['train'][e], self.epoch_losses['val'][e])) 
            # Save Model every epoch
            if self.save_model and (e+1)%5 == 0:
                self.model.save(e, run_identifier, self.epoch_losses, self.global_losses_100, run_identifier, args, True)
            
    # forward encoder & decoder
    def forward(self, batch, e, b):
        labelsHz = batch[1]     #Hz values
        batch = batch[0]        # x & y values 

        rand_idx = np.random.randint(0, batch.shape[0])
        
        batch = batch.float()
        tensor_labels = self.getLabel(labelsHz) 
        if self.cuda == True and self.device.type == 'cuda':     
            batch = batch.cuda()
            tensor_labels = tensor_labels.cuda()

        _is_training = self.model.network.training
        dset = 'train' if self.model.network.training else 'val'

        z_all, mean, logvar = self.model.network.encode(batch, cat_output=False) 
        # z_all is a list, consists of two tensors z2,z1
        
        #Display z just once every epoch 
        if b == 0 and self.plot:
            self.batch_to_color(z_all[0], f"MicroMacroZ", f"epoch{e}-z2")
            self.batch_to_color(z_all[1], f"MicroMacroZ", f"epoch{e}-z1")
        
        # Divide 64 features of z_all[0] and z_all[1] -> 128 features in 80% bzw 20%
        size_w = int(0.2*z_all[0].shape[1]) + 1         #von 64 features: 13  
        size_z = int(0.8*z_all[0].shape[1])             #von 64 features: 51
        z2_w, z2_z = torch.split(z_all[0], [size_w, size_z], dim = 1)    #torch.Size([bs = 32, 13]) torch.Size([bs, 51])
        z1_w, z1_z = torch.split(z_all[1], [size_w, size_z], dim = 1)    #torch.Size([bs, 13]) torch.Size([bs, 51])
        z = cat([z2_z, z1_z], 1)    # torch.Size([bs, 102]) 2*51 = 2*size_z
        
        out_decX, destroyedBatch = self.model.network.decoder(z_all, batch, is_training=_is_training) 
        #Display destroyed input batch from decoder for a certain, randomly chosen, batch
        # same random batch as sample & sample_rec
        if dset == 'train' and b == 0 and self.plot:
            self.batch_to_color(destroyedBatch[rand_idx,:,:-1], f"XandY", f"epoch{e}-destroyed batch")
        
        out_adversary = self.model.network.adversary_decoder(z) 
        
        #loss = MSE(output, target)
        reconstructed_batch = out_decX
        # darf nur encoder und decoder_x anpassen dürfen, nicht adversary
        loss_decX = self.loss_fn(reconstructed_batch, batch).reshape(reconstructed_batch.shape[0], -1).sum(-1).mean()
        
        self.running_loss[dset] += loss_decX.item()
        self.currentLoss['MSE'] = loss_decX
        
        #loss = CEL, label - output [self.hz]
        #tensor_labels for 500Hz: 4 (index) (0 0 0 0 1 0 (classes: 30 60 120 250 500 1000))
        loss_adv = self.loss_adv(out_adversary, tensor_labels)     
        
        self.running_loss[str('CEL' + dset)] += loss_adv.item()
        self.currentLoss['CEL'] = loss_adv        

        #update network if we are training
        if self.model.network.training:
            self.running_loss_100[dset] += loss_decX.item()
            self.running_loss_100[str('CEL' + dset)] += loss_adv.item()
           
            # Hier adv_loss minimieren
            self.model.optim_basis.zero_grad()
            loss_combo = loss_decX - (loss_adv*10)
            loss_combo.backward()
            self.model.optim_basis.step()       

        return batch[rand_idx].cpu(), reconstructed_batch[rand_idx].cpu()

        ############################################################################################################### 
        # forward just for adverserial part
    def forward_adv(self, batch):
        labelsHz = batch[1]     # Hz values
        batch = batch[0]        # x & y values

        batch = batch.float()
        tensor_labels = self.getLabel(labelsHz) 
        if self.cuda == True and self.device.type == 'cuda': 
            batch = batch.cuda()
            tensor_labels = tensor_labels.cuda()

        dset = 'train' if self.model.network.training else 'val'

        #ENCODING
        z_all, mean, logvar = self.model.network.encode(batch, cat_output=False) 
 
        size_w = int(0.2*z_all[0].shape[1]) + 1          
        size_z = int(0.8*z_all[0].shape[1])             
        z2_w, z2_z = torch.split(z_all[0], [size_w, size_z], dim = 1)    
        z1_w, z1_z = torch.split(z_all[1], [size_w, size_z], dim = 1)    
        z = cat([z2_z, z1_z], 1)    
                                   
        out_adversary = self.model.network.adversary_decoder(z)
        
        loss_adv = self.loss_adv(out_adversary, tensor_labels) 
        
        #update network if we are training
        if self.model.network.training:
            # Hier adv_loss minimieren
            self.model.optim_adversarial.zero_grad()
            # compute gradients of the params wrt the loss
            loss_adv.backward()
            # update all the params by substracting the gradients
            self.model.optim_adversarial.step()


    def evaluate_representation(self, sample, sample_rec, i, tensorboard_acc, do_eval):
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

        if do_eval:
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
    
    
    #Returns the index/Class of the Label
    def getLabel(self,  tensorHz): 
        def getIndx(Hz):        
            possibleHz = [30,60,120,250,500,1000,0]
            indx = possibleHz.index(Hz)
            return indx
        
        labelasclasses = np.array([list(map(lambda z: getIndx(z), x1)) for x1 in tensorHz])
        labelasclasses = labelasclasses[:,0]
        
        assert not labelasclasses.__contains__(6), 'train_adv-getLabel: Class 6 appeared - just a padded trial? corpus.py - iter_clice_chunks. Just skip this batch then?'
        return torch.tensor(labelasclasses, dtype=torch.long)
    

    #batch is a random chosen batch [rand_inx, 2, 1000] (bs = 32: [32,2,1000])
    #directory either 'MicroMacroZ' or 'XandY'
    def batch_to_color(self, batch, directory, name):
        
        title = 'default'

        # name: e.g. train-epoch{e}-original-batch
        if directory == 'XandY':
            scale_min = 0
            scale_max = 3
            fig = plt.figure(figsize=(10,5))
            
            if torch.is_tensor(batch): # for destroyed batch
                x = pd.DataFrame(batch[0,:].cpu().detach().numpy().reshape(-1,50))
                y = pd.DataFrame(batch[1,:].cpu().detach().numpy().reshape(-1,50))
            else:
                x = pd.DataFrame(batch[0,:].reshape(-1,50).numpy())
                y = pd.DataFrame(batch[1,:].reshape(-1,50).numpy())
            lista = [x,y]

            #---------------------------
            if self.signal_type == 'vel':
                title = 'vel_'
            else:
                title = ''

            grid = ImageGrid(fig, 111,          
                 nrows_ncols=(2,1),
                 axes_pad=0.3,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="4%",
                 cbar_pad=0.15,
                 )

            # Add data to image grid
            for i,ax in enumerate(grid):
                im = ax.imshow(lista[i], vmin=scale_min, vmax=scale_max)
                if i == 0:
                    nameCoord = 'x'
                else:
                    nameCoord = 'y'
                ax.set_title(f"{title}{nameCoord}")
                ax.grid(False)

            # Colorbar
            ax.cax.colorbar(im, extend = 'both', extendfrac = 0.2, extendrect = True)
            ax.cax.toggle_label(True)
            ax.set_xticks([])       #no axis shown
            ax.set_yticks([])
            fig.savefig(f"batchDisplay/{directory}/{name}.png")

            #---------------------------------without scaling
            fig,axs = plt.subplots(2)
            im1 = axs[0].matshow(x)
            fig.colorbar(im1, ax = axs[0], orientation = 'vertical')
            axs[0].set_title(f"{title}x", pad = 30)

            im2 = axs[1].matshow(y)
            fig.colorbar(im2, ax = axs[1], orientation = 'vertical')
            axs[1].set_title(f"{title}y", pad = 30)           
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(f"batchDisplay/{directory}/withoutScaling/{name}.png")
        #name: epoch{e}-z2
        elif directory == 'MicroMacroZ':
            scale_min = -5
            scale_max = 20
            fig = plt.figure(figsize=(10,5))
            
            x = pd.DataFrame(batch.cpu().detach().numpy())
            
            im = plt.imshow(x,cmap='cool', vmin=scale_min, vmax=scale_max)
            plt.xlabel('features')
            plt.ylabel('batches')
            plt.title(f"{name}")
            plt.grid(False)
            plt.colorbar(im, extend = 'both', extendfrac = 0.2, extendrect = True)
            plt.xticks(list(range(0,64,5)))       
            plt.yticks(list(range(0,32,5)))
            fig.savefig(f"batchDisplay/{directory}/{name}.png")

            #-----------------------WithoutScaling
            fig, axs = plt.subplots(1)
            #x = pd.DataFrame(batch.cpu().detach().numpy())
            im1 = axs.matshow(x, cmap='cool')
            fig.colorbar(im1, ax = axs, orientation = 'vertical')
            axs.set_title(f"{name}", pad=30)   
            plt.grid(None)
            plt.savefig(f"batchDisplay/{directory}/withoutScaling/{name}.png") 
        

    def batch_diff_to_color(self, original, rec, directory, name):
        diff = torch.subtract(original, rec)

        self.batch_to_color(torch.abs(diff), directory, f'{name}_diff')


args = get_parser().parse_args()
run_identifier = f"ADV-SyBa_{args.hz}Hz_{args.signal_type}_test"
setup_logging(args, run_identifier)
print_settings()

logging.info('\nRUN ADVERSARIAL: ' + run_identifier + '\n')
print('Arguments ', args)

adversarial = CSVAE()
adversarial.train()
