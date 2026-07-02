import os
import time
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import init_model
from utils import CustomDataset, AdaptiveMarginLoss, InvalidArgument, ContrastiveMarginLoss, prepare_transforms
from train import train_softmax, train_triplet, train_contrastive
from evaluate import eval_softmax, eval_triplet

def run(args):
    exp_start = time.time() 
    
    #Get the Experiment ID if it is not indicated manually.
    if args.exp is None:
        if os.path.exists(args.resultpath):
            results = pd.read_csv(args.resultpath)
            exp_no = results['exp'].iloc[-1]+1
        else:
            exp_no = 0
    else:
        exp_no = args.exp
        
    #Define if the validation set is exist or not
    if args.val:
        train_val = ['train', 'val']
        train_val_test = ['train', 'val', 'test']
    else:
        train_val = ['train']
        train_val_test = ['train', 'test']
    
    
    transform_lists = prepare_transforms(args)
    data_transforms = {x : transforms.Compose(transform_lists[x]) for x in train_val_test}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on', device)
    for fold in args.folds:
        
        print(f'Fold {fold}:')
        
        #Initialize model
        model, out_features = init_model(args.model, device, args.fs)
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path))
        #Create datasets and dataloaders
        dataset = {x: CustomDataset(x, fold, args.classes, args.dim, args.foldpath, args.datapath, data_transforms[x])
                       for x in train_val_test}
        bs = 1 if args.dim == '3d' else args.batchsize
        dataloader = {x: DataLoader(dataset[x], batch_size=bs, shuffle=True,  pin_memory=True) 
                      for x in train_val_test}
        dataset_sizes = {x: len(dataset[x]) for x in train_val_test}
        class_weights = dataset['train'].class_weights
        #Create optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
        step = 5 if args.dlr else args.epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs/step)
        
        neighbors = [9]
        #Train and Evaluate
        if args.mode == 'triplet':

            if args.adaptive:
                criterion = AdaptiveMarginLoss()
            else:
                criterion = nn.TripletMarginLoss(margin=args.margin)
            model, tr_acc, tr_auc = train_triplet(args.diff, args, exp_no, model, dataloader, optimizer, scheduler, criterion,
                                                  dataset_sizes, device, train_val, out_features)
            #torch.save(model.state_dict(), f'./models/E{exp_no}_F{fold}_triplet.pth')
            eval_triplet(exp_no, fold, args, model, dataloader, device, dataset_sizes, train_val_test, neighbors)

        elif args.mode == 'softmax':

            criterion = nn.CrossEntropyLoss(torch.tensor(class_weights).to(device))
            model, tr_acc, tr_auc = train_softmax(args, exp_no, model, dataloader, optimizer, scheduler, criterion,
                                                  dataset_sizes, device, train_val)
            #torch.save(model.state_dict(), f'./models/E{exp_no}_F{fold}_softmax.pth')

            eval_softmax(exp_no, fold, args, model, dataloader['test'], device, dataset_sizes,  tr_acc, tr_auc)

        elif args.mode == 'contrastive':
            criterion = ContrastiveMarginLoss(scale=1000)
            model, tr_acc, tr_auc = train_contrastive(args.diff, args, exp_no, model, dataloader, optimizer, scheduler, criterion, 
                                                  dataset_sizes, device, train_val, out_features)
            eval_triplet(exp_no, fold, args, model, dataloader, device, dataset_sizes, train_val_test, neighbors)
            
        else:
            raise InvalidArgument(f'{args.mode} is not a valid mode! Please use triplet, contastive or softmax.')
          
    time_elapsed = time.time() - exp_start
    print(f'Experiment complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    

if __name__ == '__main__': 
    #CUDA_LAUNCH_BLOCKING=1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--foldpath',  help='Path to folds')
    parser.add_argument('-dp', '--datapath', nargs='*',  help='Path to data')
    parser.add_argument('-rp', '--resultpath', help='Path to results')
    parser.add_argument('-exp', default=100, help='Experiment ID')
    parser.add_argument('-dim', default='2d', help='Input dimension')
    parser.add_argument('-model', default='shufflenet', help='Name of the model')
    parser.add_argument('-mp', '--model_path', help='Path of the model')
    parser.add_argument('-fs', type=int, help='Output feature size')
    parser.add_argument('-val', action='store_true', help='validation')
    parser.add_argument('-dlr', action='store_true', help='Decrease learning rate')
    parser.add_argument('-lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('-m', '--margin', default=0.2, type=float, help='Triplet loss margin')
    parser.add_argument('-e', '--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('-bs', '--batchsize', default=32, type=int, help='Batch size')
    parser.add_argument('-mode', help='triplet or softmax?')
    parser.add_argument('-folds', default=[0,1,2,3,4], help='which folds to experiment')
    parser.add_argument('-temp', action='store_true')
    parser.add_argument('-phase', help='train or test')
    parser.add_argument('-triplets', default=1, type=int, help='Number of triplets')
    parser.add_argument('-semi', action='store_true', help='semi triplet: only ones are selected as anchor.')
    parser.add_argument('-tsm', default=1, type=int, help='triplet selection method')
    parser.add_argument('-diff', default=0.2, type=float, help='difficulty threshold')
    parser.add_argument('-adaptive', action='store_true', help='adaptive margin')
    parser.add_argument('-classes', default=2, type=int, help='number of classes')
    parser.add_argument('-bc', '--binary_conversion', help='Binary conversion: before or after evaluation')
    parser.add_argument('-slices', default='tumor', help='slice type: tumor or lung?')
    #batchsize, epochs, folds, mode, fs
    args = parser.parse_args()
    
    run(args)

