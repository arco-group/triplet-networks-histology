import copy
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from utils import plot_graphs, InvalidArgument
from triplet_selection import tsm0, tsm1, tsm2, tsm3, tsm4

def train_softmax(args, exp_no, model, dataloader, optimizer, scheduler, criterion, dataset_sizes, device, train_val):
    training_start = time.time()
    print('Training started.')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    loss_list = {x: [] for x in train_val}
    acc_list = {x: [] for x in train_val}
    auc_list = {x: [] for x in train_val}
    for epoch in range(args.epochs):
        epoch_start = time.time()
        y_true, y_score = {}, {}
        print(f'----- Epoch {epoch+1}/{args.epochs} -----')
        for phase in train_val:
            
            running_loss = 0
            running_corrects = 0
            y_true[phase] = torch.Tensor().to(device)
            y_score[phase] = torch.Tensor().to(device)
            for data in dataloader[phase]:
                inputs, labels = data[0].to(device), data[1].to(device)
                if args.slices == 'mnist':
                    labels = torch.where(labels<5, 0, 1)
                optimizer.zero_grad()
                if phase == 'train':
                    model.train(True)
                    outputs = model(inputs)   
                    
                else:
                    model.train(False)
                    with torch.no_grad():
                        outputs = model(inputs)

                loss = criterion(outputs, labels)
                
                preds = torch.argmax(outputs, dim=1)

                running_loss += loss.item() * inputs.shape[0]
                running_corrects += torch.sum(preds == labels).cpu()
                y_true[phase] = torch.cat((y_true[phase], labels))

                y_score[phase] = torch.cat((y_score[phase], F.softmax(outputs-torch.max(outputs, dim=1)[0].unsqueeze(dim=1), dim=1)))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if args.classes == 2:
                epoch_auc = roc_auc_score(y_true[phase].detach().cpu().numpy(), y_score[phase].detach().cpu().numpy()[:,1])
            else: 
                if args.slices == 'mnist':
                    epoch_auc = 0
                else:
                    epoch_auc = 0
            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc)
            auc_list[phase].append(epoch_auc)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
               
        time_elapsed = time.time() - epoch_start
        print(f'Epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    name = f'E{exp_no}_train'
    plot_graphs(loss_list, acc_list, auc_list, name, train_val)
    time_elapsed = time.time() - training_start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    if args.val:
        model.load_state_dict(best_model_wts)
    return model, float(epoch_acc), epoch_auc

def train_triplet(diff_th, args, exp_no, model, dataloader, optimizer, scheduler, criterion, dataset_sizes, device, train_val, out_features):
    training_start = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = {x: [] for x in train_val}
    max_iter = {x: int(dataset_sizes[x]/args.batchsize) for x in train_val}
    last_batch = {x: dataset_sizes[x]%args.batchsize for x in train_val}
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f'----- Epoch {epoch+1}/{args.epochs} -----')
        for phase in train_val:
            features = torch.FloatTensor().to(device)
            labels = torch.FloatTensor().to(device)
            k = 0
            c = 0
            running_loss = 0
            for data in dataloader[phase]:
                if args.slices in ['mnist', 'cifar10']:
                    inputs, label, survivals = data[0].to(device), data[1].type(torch.FloatTensor).to(device), 0
                    label = torch.where(label<5, 0, 1)
                else:
                    inputs, label, survivals = data[0].to(device), data[1].type(torch.FloatTensor).to(device), data[2].to(device)

                optimizer.zero_grad()
                k += inputs.shape[0]
                
                if phase == 'train':
                    model.train(True)
                    outputs = model(inputs)   
                    
                else:
                    model.train(False)
                    with torch.no_grad():
                        outputs = model(inputs)

                outputs = F.normalize(outputs.view(outputs.shape[0], out_features))
                features = torch.cat((features, outputs))
                labels = torch.cat((labels, label))
                
                if ((c != max_iter[phase]) & (k == args.batchsize)) | ((c == max_iter[phase]) & (k == last_batch[phase])):
                    
                    distances = torch.cdist(features, features)
                    indices = torch.argsort(distances)

                    if args.tsm == 4:
                        loss = tsm4(args.margin, distances, labels)
                        running_loss += loss.item()*k
                        if phase == 'train':
                            loss.backward()
                    else:
                        for i,lbl in enumerate(labels):
                            if args.semi:
                                if labels[i] == 0:
                                    continue
                           
                            positives = indices[i][labels[indices][i] == lbl][1:]
                            negatives = indices[i][labels[indices][i] != lbl]
                            if len(positives)==0 or len(negatives)==0:
                          
                                continue
                            if args.tsm == 0:
                                loss = tsm0(args, criterion, distances[i], positives, negatives, features, i, out_features, device)
                            elif args.tsm == 1:
                                loss = tsm1(args, criterion, positives, negatives, features, survivals, i, out_features)
                            elif args.tsm == 2:
                                loss = tsm2(args, diff_th, criterion, positives, negatives, features, i, out_features, phase)
                            elif args.tsm == 3:
                                loss = tsm3(args, diff_th, criterion, positives, negatives, features, i, out_features, phase)
                            else:
                                raise InvalidArgument(f'{args.tsm} is not a valid selection method! Please choose among 1,2 or 3.')
                        
                            if loss == -1:
                                continue
                            
                            running_loss += loss.item()
        
                            if phase == 'train':
                                loss.backward(retain_graph=True)
                         
                    if phase == 'train':
                        optimizer.step()
                    del features
                    del labels
                    features = torch.FloatTensor().to(device)
                    labels = torch.FloatTensor().to(device)
                    k = 0
                    c += 1  
            if phase == 'train':    
                scheduler.step() 
            epoch_loss = running_loss/dataset_sizes[phase]
            loss_list[phase].append(epoch_loss)
            
            print(f'{phase} loss: {epoch_loss}')
        time_elapsed = time.time() - epoch_start
        print('Epoch completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    time_elapsed = time.time() - training_start
    training_time = str(time_elapsed // 60) + 'm ' + str(round(time_elapsed % 60)) + 's'
    print(f'Training completed in {training_time}')
    
    return model, 0, 0

def train_contrastive(diff_th, args, exp_no, model, dataloader, optimizer, scheduler, criterion, dataset_sizes, device, train_val, out_features):
    training_start = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = {x: [] for x in train_val}
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f'----- Epoch {epoch+1}/{args.epochs} -----')
        for phase in train_val:
            
            running_loss = 0
            for data in dataloader[phase]:
                inputs, labels, survivals = data[0].to(device), data[1].type(torch.FloatTensor).to(device), data[2].to(device)
                optimizer.zero_grad()
                if phase == 'train':
                    model.train(True)
                    outputs = model(inputs)   
                    
                else:
                    model.train(False)
                    with torch.no_grad():
                        outputs = model(inputs)
                
                distances = torch.cdist(outputs, outputs)
                indices = torch.argsort(distances)
                loss = criterion(distances, survivals)
                running_loss += loss.item()*inputs.shape[0]
                    
                    
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
            if phase == 'train':    
                scheduler.step() 
            epoch_loss = running_loss/dataset_sizes[phase]
            loss_list[phase].append(epoch_loss)
            
            print(f'{phase} loss: {epoch_loss}')
        time_elapsed = time.time() - epoch_start
        print('Epoch completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(f'Min: {torch.min(outputs)}, Max: {torch.max(outputs)}, Mean:{torch.mean(outputs)}, Std:{torch.std(outputs)}')
                                                
    time_elapsed = time.time() - training_start
    training_time = str(time_elapsed // 60) + 'm ' + str(round(time_elapsed % 60)) + 's'
    print(f'Training completed in {training_time}')
                    
    return model, 0, 0