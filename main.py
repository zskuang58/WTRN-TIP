from option import args
from utils import  mkExpDir, set_random_seed
from dataset import dataloader
from importlib import import_module
from waveloss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)
    set_random_seed(args.seed)
    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    WTRN = import_module( 'models.' + args.which_model + '.WTRN')
    _model = WTRN.WTRN(args).to(device)
    # _model = TTSR.TTSR(args).to(device)
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        # print(t)
        # exit()
        t.evaluate()
    else:
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            if (args.num_init_epochs == 0) & (args.adv_w == 0):
                is_init = True
            else: 
                is_init = False
            t.train(current_epoch=epoch, is_init=is_init)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
