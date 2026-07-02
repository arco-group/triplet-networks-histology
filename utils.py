import os
import csv
import nrrd
import pickle
import numpy as np
import pandas as pd
#import torchio as tio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class InvalidArgument(Exception):
    pass


class BSTripletLoss(object):
    def __call__(self, batch):
        loss = 0
        return loss
        
    
class ContrastiveMarginLoss(object):
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, distance, survival):
        survival = survival.view(-1,1)
        margin = abs(survival-torch.transpose(survival,0,1))/self.scale
        loss = abs(distance - margin)
        loss = torch.mean(loss)
        return loss
    
    
    
class AdaptiveMarginLoss(object):
    
    def __call__(self,a,p,n,m):
        pdist = nn.PairwiseDistance(p=2)
        dp = pdist(a,p)
        dn = pdist(a,n)
        cost = dp-dn+m
        loss = torch.where(cost > 0, cost, torch.zeros_like(cost))
        loss = torch.mean(loss)
        return loss

class CustomDataset(Dataset):
    
    def __init__(self, phase, fold, class_no, dim, foldpath, datapath, transform=None):
        self.transforms = transform
        self.dim = dim
        self.datapath = datapath
        
        df = pd.read_excel(io=foldpath, sheet_name=f'_fold{fold}_{phase}', keep_default_na=False)
        try:
            self.id_list = df['id']
        except KeyError:
            self.id_list = df['PatientID']
        try:
            self.label_list = df['y'] if class_no == 2 else df[f'y_{class_no}']
        except KeyError:
            self.label_list = df['label']
        try:
            self.survival_list = df['survival']
        except KeyError:
            self.survival_list = df['y']#df['Survival.time']

        self.class_weights = []
        for i in range(self.label_list.nunique()):
            self.class_weights.append(1-(len(self.label_list[self.label_list == i])/len(self.label_list)))

        if self.dim == '1d':
            with open('/Users/fatih/My Drive/UCBM/humanitas/selected_features.pkl', 'rb') as f:
                feature_list = pickle.load(f)
            self.features = pd.read_csv(datapath)
            ID = self.features['PatientID']
            self.features = self.features[feature_list[fold]]
            self.features.drop(columns=['PatientID', 'label', 'Unnamed: 0'], inplace=True, errors='ignore')
            
            
            self.features = (self.features-self.features.mean())/(self.features.std()+1e-15)
            self.features['PatientID'] = ID
            
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx): #Should be changed according to dimensions
        if self.dim == '3d':
            filename = str(self.id_list.iloc[idx])+'_croppedimage.nrrd'
            filepath = os.path.join(self.datapath, filename)
            pid = self.id_list.iloc[idx]
            img, _ = nrrd.read(filepath)
        elif self.dim == '2d':

            try:
                #AERTS 0
                filepath = os.path.join(self.datapath[0], self.id_list.iloc[idx].split('_')[0], self.id_list.iloc[idx])
                img, _ = nrrd.read(filepath)
                pid = self.id_list.iloc[idx].split('_')[0]
            except:
                #HUMANITAS 1
                filepath = os.path.join(self.datapath[1], '_'.join(self.id_list.iloc[idx].split('_')[:-1]), self.id_list.iloc[idx])
                img, _ = nrrd.read(filepath)
                pid = '_'.join(self.id_list.iloc[idx].split('_')[:-1])
            img = np.clip(img, -600, 400)
            img = (img+600)/1000

        elif self.dim == '1d':
            img = self.features[self.features['PatientID']==self.id_list.iloc[idx]]
            img.drop(columns=['PatientID'], inplace=True)
            img = img.to_numpy(np.float32).reshape((1,-1))
            pid = self.id_list.iloc[idx]

            
        label = self.label_list[idx]
        survival_time = self.survival_list[idx]

        if self.transforms is not None:
            img = self.transforms(img.astype(np.float32))
            if self.dim == '1d':
                img = torch.squeeze(img)
 
        return img, label, survival_time, pid
    
def prepare_transforms(args):
    #Transformations
    
    if args.dim == '3d':
        transform_lists = {'train': [transforms.ToTensor(),
                                     transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)), 
                                     tio.ZNormalization()],
                          'val':   [transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)), 
                                    tio.ZNormalization()],
                          'test':  [transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)), 
                                    tio.ZNormalization()]}
        
    elif args.dim == '2d':
        if args.slices == 'tumor':
            transform_lists = {'train': [transforms.ToTensor(),
                                         #transforms.Resize((64,64)), 
                                         transforms.RandomHorizontalFlip(),
                                         #transforms.RandomVerticalFlip(),
                                         transforms.RandomRotation(degrees=(-5,5)),
                                         #transforms.Normalize(-342,277)
                                         ],
                              'val':   [transforms.ToTensor(),
                                        #transforms.Resize((64,64)),
                                        #transforms.Normalize(-342,277)
                                        ],
                              'test':  [transforms.ToTensor(),
                                        #transforms.Resize((64,64)),
                                        #transforms.Normalize(-342,277)
                                        ]}
            # Crop instead of resize, 128 is chosen to contain entire tumor. playground/patch_dimensions.ipynb
            if os.path.split(args.datapath[0])[1][-2:] != '64':
                for x in ['train', 'val', 'test']:
                    #transform_lists[x].append(transforms.Resize((64,64)))
                    transform_lists[x].append(transforms.CenterCrop(128))

        elif args.slices == 'lung':
            transform_lists = {'train': [transforms.ToTensor(),
                                         transforms.Resize((120,240)), 
                                         transforms.Normalize(-363, 417)],
                              'val':   [transforms.ToTensor(),
                                        transforms.Resize((120,240)),
                                        transforms.Normalize(-363, 417)],
                              'test':  [transforms.ToTensor(),
                                        transforms.Resize((120,240)),
                                        transforms.Normalize(-363, 417)]}
        elif args.slices == 'mnist':
            transform_lists = {'train': [transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))],
                              'val':   [transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))],
                              'test':  [transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))]}
        elif args.slices == 'cifar10':
            transform_lists = {'train': [transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                              'val':   [transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                              'test':  [transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]}
        
        else:
            raise InvalidArgument(f'{args.slices} is not a valid slice type! Options are tumor or lung.')
            
    elif args.dim == '1d':
        transform_lists = {'train': [transforms.ToTensor()],
                          'val':   [transforms.ToTensor()],
                          'test':  [transforms.ToTensor()]}
    
    if args.model in ['resnet50','mobilenet', 'vgg16', 'googlenet', 'densenet', 'shufflenet'] and args.slices != 'cifar10':
        for x in ['train', 'val', 'test']:
            transform_lists[x].append(transforms.Lambda(lambda x: x.repeat(3,1,1)))
    
    return transform_lists

def plot_graphs(loss, acc, auc, name, sets):
  fig = plt.figure(figsize=(30,10))
  ax0 = fig.add_subplot(131, title="loss")
  ax1 = fig.add_subplot(132, title="acc")
  ax2 = fig.add_subplot(133, title="auc")
  ax0.set_yscale('log')
  colors = ['bo-', 'ro-', 'go-']
  for i, phase in enumerate(sets):
      ax0.plot(loss[phase], colors[i], label=phase)
      ax1.plot(acc[phase], colors[i], label=phase)
      ax2.plot(auc[phase], colors[i], label=phase)
  ax0.legend()
  ax1.legend()
  ax2.legend()
  path = f'./results/{name}_graphs.jpg'
  fig.savefig(path)
  print(f'Figure is saved at {path}')
  plt.close()
  
def plot_feature_map(features, labels, exp_no, fold):
    tsne = TSNE(init='pca', learning_rate='auto') 
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for i,phase in enumerate(['train', 'test']):
        features_transformed = torch.from_numpy(tsne.fit_transform(features[phase].cpu()))
        for g in np.unique(labels[phase].cpu()):
            ix = np.where(labels[phase].cpu() == g)
            axs[i].scatter(features_transformed[ix,0].cpu(), features_transformed[ix,1].cpu(), label=g)
        axs[i].legend()
        axs[i].set_title(phase)
    fname = f'./results/E{exp_no}_F{fold}_feature_map.jpg'
    plt.savefig(fname)
    print(f'Feature plot is saved at {fname}')
    plt.close()
        
        
def write_results(args, exp_no, fold, tr_acc, tr_auc, results, nn, p):
    eps = 1e-15
    tn, fp, fn, tp = results['tn'], results['fp'], results['fn'], results['tp']
    sensitivity = tp/(tp+fn+eps)
    specificity = tn/(tn+fp+eps)
    precision = tp/(tp+fp+eps)
    recall = sensitivity
    f1 = 2*precision*recall/(precision+recall+eps)
    gmean = np.sqrt(sensitivity*specificity)
    is_file = os.path.isfile(args.resultpath)
    with open(args.resultpath, 'a', newline='') as csvfile:
      fieldnames = ['exp', 'fold', 'mode', 'model', 'epochs', 'batch size', 'lr', 'reduce lr',
                    'classes', 'triplets', 'nn', 'tsm', 'semi', 'patch/patient',
                    'train_acc', 'train_auc', 'test_acc', 'test_auc', 'tn', 'fp', 'fn', 'tp',
                    'sensitivity', 'specificity', 'precision', 'recall', 'f1', 'gmean']
      
      logger = csv.DictWriter(csvfile, fieldnames=fieldnames)
      if not is_file:
          logger.writeheader()
      logger.writerow({ 'exp': exp_no,   
                        'fold': fold,
                        'mode': args.mode,
                        'model': args.model,
                        'epochs': args.epochs,
                        'batch size': args.batchsize,
                        'lr': args.lr,
                        'reduce lr': args.dlr,
                        'classes': args.classes,
                        'triplets': args.triplets,
                        'nn': nn,
                        'tsm': args.tsm,
                        'semi': args.semi,
                        'patch/patient': p,
                        'train_acc': tr_acc,
                        'train_auc': tr_auc,
                        'test_acc' : results['acc'],
                        'test_auc': results['auc'],
                        'tn' : results['tn'],
                        'fp' : results['fp'],
                        'fn' : results['fn'],
                        'tp' : results['tp'],
                        'sensitivity' : sensitivity,
                        'specificity' : specificity,
                        'precision' : precision,
                        'recall' : recall,
                        'f1' : f1,
                        'gmean' : gmean})
    