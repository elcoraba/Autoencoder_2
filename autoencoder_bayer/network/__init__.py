import logging
import torch

from .autoencoder import TCNAutoencoder
from .supervised import SupervisedTCN

import json
from datetime import datetime, date


class ModelManager:
    def __init__(self, args, training=True, **kwargs):
        self.is_training = training
        self.load_network(args, **kwargs)

    def load_network(self, args, **kwargs):
        if args.model_pos or args.model_vel:
            print('Load network')
            if self.is_training:
                self.network, self.optim = self._load_pretrained_model(args.model_pos or args.model_vel)
            else:
                self.network = {}
                vel_net = self._load_pretrained_model(args.model_vel)
                if vel_net:
                    self.network['vel'] = vel_net.eval() # packe key in dict, was netzwerk enthält, netzwerk in eval modus versetzen (sind nicht im traiuning)
                pos_net = self._load_pretrained_model(args.model_pos)
                if pos_net:
                    self.network['pos'] = pos_net.eval()
        #erstelle Netzwerk
        else:
            self.network = TCNAutoencoder(args)

            
            #-B---
            params_basis = [*self.network.encoder.parameters(), *self.network.decoder.parameters()]
            params_adversarial = [*self.network.adversary_decoder.parameters()]
            
            self.optim_basis = torch.optim.Adam(params_basis, lr=args.learning_rate)
            self.optim_adversarial = torch.optim.Adam(params_adversarial, lr=args.learning_rate)
            #-B---
            self._log(self.network)

    def _load_pretrained_model(self, model_name):
        if not model_name:
            return None

        logging.info('Loading saved model {}...'.format(model_name))
        model = torch.load('../models/' + model_name)
        try:
            network = model['network']
        except KeyError:
            network = model['model']
        self._log(network)

        network.load_state_dict(model['model_state_dict'])

        if self.is_training:
            optim = torch.optim.Adam(network.parameters())
            optim.load_state_dict(model['optimizer_state_dict'])
            return network, optim

        return network

    def _log(self, network):
        logging.info('\n ' + str(network))
        logging.info('# of Parameters: ' +
                     str(sum(p.numel() for p in network.parameters()
                             if p.requires_grad)))

    def save(self, e, run_identifier, losses, losses_100, name_run, args, is_adv=False): 
        model_filename = '../models/' + args.signal_type + '-e' + str(e) + '-hz' + str(args.hz) #pos-i3738, i = iteration
        torch.save(
            {
                'epoch': e,
                'network': self.network,
                'model_state_dict': self.network.state_dict(), #a dictionary containing a whole state of the module, contains weights
                'optimizer_state_dict': self.optim_basis.state_dict(),
                'optimizer_state_dict_adv': self.optim_adversarial.state_dict() if is_adv else [], 
                'losses': losses
            }, model_filename)
        logging.info('Model saved to {}'.format(model_filename))

        #save params in extra file
        params = {
            "run_identifier": run_identifier,
            "signal_type"   : args.signal_type,               
            "lr"            : args.learning_rate,
            "hz"            : args.hz,
            "viewing time"  : args.viewing_time,
            "bs"            : args.batch_size,      #I added
            "slice-time-windows": args.slice_time_windows,
            "last loss"     : list(losses)[-1],     #before: losses[-1],
            "current day"   : date.today().strftime("%d.%m.%Y"),
            "current time"  : datetime.now().strftime("%H:%M:%S"),
            "epoch losses train"  : list(losses['train']),
            "epoch losses val"  : list(losses['val']),
            "global losses 100 train" : list(losses_100['train']),
            "global losses 100 val": list(losses_100['val']),          # losses every 100 batches
            "CEL epoch losses train"  : list(losses['CELtrain']) if is_adv else [],
            "CEL epoch losses val"  : list(losses['CELval']) if is_adv else [],
            "CEL global losses 100 train" : list(losses_100['CELtrain']) if is_adv else [],
            "CEL global losses 100 val": list(losses_100['CELval'] if is_adv else []) 
        }

        jsonFile = json.dumps(params) #was .dump()

        with open(model_filename + '.json', 'w') as jasonfile: # function opens a file, and returns it as a file object., Write - Opens a file for writing, creates the file if it does not exist
            jasonfile.write(jsonFile)

    '''
    def save(self, i, run_identifier, losses):
        model_filename = '../models/' + run_identifier + '-i' + str(i)
        torch.save(
            {
                'iter': i,
                'network': self.network,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'losses': losses
            }, model_filename)
        logging.info('Model saved to {}'.format(model_filename))
    '''
