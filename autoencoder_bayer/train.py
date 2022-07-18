#tensorboard --logdir=runs
import time
import logging
from datetime import datetime

import numpy as np
from torch import manual_seed, nn, device, cuda, multiprocessing, cat
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


class Trainer:
    def __init__(self):
        self.model = ModelManager(args)

        self.save_model = args.save_model
        self.cuda = args.cuda
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        if self.cuda and self.device.type == 'cuda':
            self.model.network = self.model.network.cuda()
        else:
            self.model.network = self.model.network.to(self.device)

        self.rec_loss = args.rec_loss #MSE
        self.epochs = args.epochs

        #-B----
        # introduced two summary writer, so we can see the two functions in one graph in tensorboard
        self.tensorboard_train = SummaryWriter(f"runs/{run_identifier}_trainLoss")
        self.tensorboard_val = SummaryWriter(f"runs/{run_identifier}_valLoss")
        self.tensorboard_first_epoch_train = SummaryWriter(f"runs/{run_identifier}_FIRSTepoch_trainLoss")
        self.tensorboard_first_epoch_val = SummaryWriter(f"runs/{run_identifier}_FIRSTepoch_valLoss")
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
    
    '''
    def reset_epoch_losses(self):
        self.epoch_losses = {'train': 0.0}
        #{'train': {l: 0.0 for l in self._loss_types},'val': {l: 0.0 for l in self._loss_types}}

    def init_global_losses(self, num_checkpoints):
        self.global_losses = {'train': np.zeros(num_checkpoints)}
            #{'train': {l: np.zeros(num_checkpoints) for l in self._loss_types},
            #'val': {l: np.zeros(num_checkpoints) for l in self._loss_types}}

    def update_global_losses(self, checkpoint):
        for dset in ['train', 'val']:
            self.global_losses[dset]['total'][checkpoint] = self.epoch_losses[dset]['total']
            #for l in self._loss_types:
            #    self.global_losses[dset][l][checkpoint] = self.epoch_losses[dset][l]
    '''

   #-B---- #was self.running_loss[dset]['total'] -> now self.running_loss[dset]
   #saves loss every 100 batches
   # was reset_running_loss_100
    def init_running_loss_100(self):
        self.running_loss_100 = {'train': 0.0}

    #adds up to the epoch loss
    def init_running_loss(self):
        self.running_loss = {'train': 0.0,
                             'val': 0.0}
    
    # was init_global_losses_100
    def init_epoch_losses_100(self, num_checkpoints):
        self.global_losses_100 = {
            'train': np.zeros(int(num_checkpoints * (int(len(self.train_dataloader)/TRAIN_SAVE_LOSS_EVERY_X_BATCHES)) ) +1),
            'val': np.zeros(int(num_checkpoints * (int(len(self.train_dataloader)/TRAIN_SAVE_LOSS_EVERY_X_BATCHES)) ) +1)}
       
    # was init_global_losses
    def init_epoch_losses(self, num_checkpoints):
        self.epoch_losses = {
            'train': np.zeros(num_checkpoints),
            'val': np.zeros(num_checkpoints)}

    #just used for first epoch, saves loss for every batch
    def init_currentLoss(self):
        self.currentLoss = {'train': 0.0}
    
   #-B----  

   #TODO Add early stopping?

   #TODO Add batch_to_color & batch_diff_to_color?

    def train(self):
        logging.info('\n===== STARTING TRAINING =====')
        logging.info('{} all samples, {} train batches, {} val batches'.format(len(self.dataset), len(self.train_dataloader), len(self.val_dataloader)))
        logging.info('Loss Function:' + str(self.loss_fn))

        self.init_epoch_losses(TRAIN_NUM_EPOCHS)
        self.init_epoch_losses_100(TRAIN_NUM_EPOCHS)
        self.init_currentLoss()
        counter_100 = 0

        t = tqdm(range(0, TRAIN_NUM_EPOCHS))
        for e in t:
        #while i < MAX_TRAIN_ITERS:
            self.init_running_loss()
            self.init_running_loss_100()

            self.model.network.train()
            for b, batch in enumerate(tqdm(self.train_dataloader, desc = 'Train Batches')):
                ###
                #if b == 0 and e == 0:
                #    self.tensorboard_train.add_graph(self.model.network, batch)
                ###
                sample, sample_rec = self.forward(batch)

                #-B----
                #save running loss 100 and reset it
                if (b+1) % TRAIN_SAVE_LOSS_EVERY_X_BATCHES == 0:
                    mean_loss = self.running_loss_100['train'] / TRAIN_SAVE_LOSS_EVERY_X_BATCHES
                    self.global_losses_100['train'][counter_100] = mean_loss                       #e*len(self.dataloader) + b
                    self.tensorboard_train.add_scalar(f"loss_per_{TRAIN_SAVE_LOSS_EVERY_X_BATCHES} batches", mean_loss, counter_100)
                    self.init_running_loss_100()
                    counter_100 +=1

                    '''
                    np.savetxt(f"original-batch-train", sample.numpy())
                    self.batch_to_color(sample, f"original-batch-train")
                    np.savetxt(f"reconstructed-batch-train", sample_rec.detach().numpy())
                    self.batch_to_color(sample_rec.detach(), f"reconstructed-batch-train")
                    self.batch_diff_to_color(sample, sample_rec.detach(), 'train')
                    '''
                if e < 1:
                    #self.tensorboard_first_epoch.add_scalar(f"loss in first epoch", self.currentLoss, b)
                    self.tensorboard_first_epoch_train.add_scalar(f"loss in first epoch TRAIN", self.currentLoss, b)
                '''
                if b == 0:
                    break
                '''
                #-B----

            # save the train loss of the whole epoch
            self.logB(e, 'train')

            #############################################################################
            # Validate the model
            self.model.network.eval()
            for b, batch in enumerate(tqdm(self.val_dataloader, desc = 'Val Batches')):
                # In forward NN also calcs & saves the loss 
                sample_v, sample_rec_v = self.forward(batch)
                ''' 
                np.savetxt(f"original-batch-val", sample_v.numpy())
                self.batch_to_color(sample_v, f"original-batch-val")
                np.savetxt(f"reconstructed-batch-val", sample_rec_v.detach().numpy())
                self.batch_to_color(sample_rec_v.detach(), f"reconstructed-batch-val")
                self.batch_diff_to_color(sample_v, sample_rec_v.detach(), 'val')
                '''
                if e < 1:
                    self.tensorboard_first_epoch_val.add_scalar(f"loss in first epoch VAL", self.currentLoss, b)
                '''
                if b == 0:
                    break
                '''

            #evaluation
            if (e + 1) % 1 == 0: #TODO 10
                print('---------------------Evaluation---------------------')
                self.evaluate_representation(sample, sample_rec, e, self.tensorboard_acc_train, True)
                self.evaluate_representation(sample_v, sample_rec_v, e, self.tensorboard_acc_val, False)

                
            self.logB(e, 'val') #TODO still there?
            t.set_postfix(loss = (self.epoch_losses['train'][e], self.epoch_losses['val'][e])) # print losses in tqdm bar
            # Save Model every epoch
            if self.save_model: #and (e+1)%5 == 0:
                self.model.save(e, run_identifier, self.epoch_losses, self.global_losses_100, run_identifier, args)

            # exit train loop, if early stopping says so
            '''
            stop = self.early_stopping(self.global_losses['val']['total'][e]) #current val loss!
            if stop:
                break
            '''
            

    def forward(self, batch):
        batch = batch.float()
        if self.cuda == True and self.device.type == 'cuda':     #batch = batch.to(self.device)
            batch = batch.cuda()

        _is_training = self.model.network.training
        out = self.model.network(batch, is_training=_is_training)

        dset = 'train' if self.model.network.training else 'val'

        reconstructed_batch = out[0]
        #loss = MSE(output, target)
        loss = self.loss_fn(reconstructed_batch, batch
                            ).reshape(reconstructed_batch.shape[0], -1).sum(-1).mean()
        self.running_loss[dset] += loss.item()
        self.currentLoss = loss

        #update network if we are training
        if self.model.network.training:
            self.running_loss_100[dset] += loss.item()
            loss.backward()
            #self.model.optim.step()
            #self.model.optim.zero_grad()
            self.model.optim_basis.step()
            self.model.optim_basis.zero_grad()

        rand_idx = np.random.randint(0, batch.shape[0])
        return batch[rand_idx].cpu(), reconstructed_batch[rand_idx].cpu()
    
    # added do_eval as we just need the different samples between train and val for the visualization, not the evaluation. Therefore we don't need to do the eval twice
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
        def get_mean_losses():
            try:
                iters = (len(self.train_dataloader) if dset == 'train'     
                         else len(self.val_dataloader))
            except TypeError:
                iters = 1
            return (self.running_loss[dset] / iters)

        def to_tensorboard(dset, loss):
            self.tensorboard.add_scalar(f'{dset}_loss_per_epoch', loss, e)
        
        if dset == 'train':
            tr_loss = get_mean_losses()
            # save the mean loss of this epoch during training
            self.epoch_losses[dset][e] = tr_loss
            # reset running loss for the epoch
            self.running_loss[dset] = 0.0
            if self.tensorboard_train:
                #to_tensorboard('train', tr_loss)
                self.tensorboard_train.add_scalar(f'loss_per_epoch', tr_loss, e)
        elif dset == 'val':
            val_losses = get_mean_losses()
            self.epoch_losses[dset][e] = val_losses
            self.running_loss[dset] = 0.0
            if self.tensorboard_val:
                self.tensorboard_val.add_scalar(tag = f'loss_per_epoch', scalar_value = val_losses, global_step = e)
    
    #TODO delete? does it get used somewhere
    def logO(self, i, num_train_iters, t):

        def get_mean_losses(dset):
            try:
                iters = (num_train_iters if dset == 'train'
                         else len(self.val_dataloader))
            except TypeError:
                iters = 1
            return self.epoch_losses[dset] / iters

        def stringify(losses):
            return ' '.join(['{}: {:.2f}'.format(loss.upper(), val)
                             for (loss, val) in losses.items()
                             if loss != 'total'])

        def to_tensorboard(dset, losses):
            for (loss, val) in losses.items():
                self.tensorboard.add_scalar(
                    '{}_{}_loss'.format(dset, loss), val, i)
        #-B---- TODO delete 
        a = 'Pappe'
        assert a == 'Pappe', 'We ended up in original log in train()'
        #-B----
        tr_losses = get_mean_losses('train')
        val_losses = get_mean_losses('val')

        # build string to print out
        string = '[{}/{}] TLoss: {:.4f}, VLoss: {:.4f} ({:.2f}s)'.format(
            i, MAX_TRAIN_ITERS, tr_losses['total'], val_losses['total'], t)
        string += '\n\t train ' + stringify(tr_losses)
        if val_losses['total'] > 0.00:
            string += '\n\t val ' + stringify(val_losses)
        logging.info(string)

        if self.tensorboard:
            to_tensorboard('train', tr_losses)
            if val_losses['total'] > 0.00:
                to_tensorboard('val', val_losses)


args = get_parser().parse_args()
run_identifier = f"SyBa_{args.hz}Hz_{args.signal_type}_ETRA_FIFA_EMVIC_adaptedPreprocessDVAincl" #datetime.now().strftime('%m%d-%H%M')
setup_logging(args, run_identifier)
print_settings()

logging.info('\nRUN: ' + run_identifier + '\n')
logging.info('Arguments ', str(args))

trainer = Trainer()
trainer.train()
