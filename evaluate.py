from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from utils import write_results, plot_feature_map, InvalidArgument


def eval_softmax(exp_no, fold, args, model, dataloader, device, dataset_sizes, tr_acc='N/A', tr_auc='N/A'):
    print('Testing ...')
    #Create tensors to store labels and scores
    y_true = torch.Tensor().to(device)
    y_score = torch.Tensor().to(device)
    pids = []
    #Get the batch
    for data in dataloader:
        if args.slices not in ['mnist', 'cifar10']:
            inputs, labels, pid = data[0].to(device), data[1].to(device), data[3]
            pids.extend(pid)
        else:
            inputs, labels = data[0].to(device), data[1].to(device)
        if args.slices == 'mnist':
            labels = torch.where(labels<5, 0, 1)
        #Extract features
        model.train(False)
        with torch.no_grad():
            outputs = model(inputs)
        
        #Store labels, scores and predictions
        y_true = torch.cat((y_true, labels))
        y_score = torch.cat((y_score, F.softmax(outputs, dim=1)))
        
        #print(f'output: {outputs}')
        #print(f'score: {y_score}')
        
    preds = torch.argmax(y_score, dim=1)
    #print(y_true, preds, y_score)
    #Calculate the results for both patients and patches if it is 2D
    for p in ['patches', 'patients']:
        
        #Get the patient's scores and predictions
        if p == 'patients':
            pids = np.array(pids)
            pts_scores, pts_preds, pts_labels = [], [], []
            for pt in np.unique(pids):
                pt_preds = preds[pids == pt]
                pt_pred = sum(pt_preds)/len(pt_preds)
                pts_scores.append(pt_pred)
                pts_preds.append(np.round(pt_pred))
                pts_labels.append(y_true[pids==pt][0])

            preds = np.array(pts_preds)
            y_true = np.array(pts_labels)
            y_score = np.array(pts_scores)
            dataset_sizes['test'] = len(y_true)
        else:
            preds = preds.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            y_score = y_score.detach().cpu().numpy()[:,1]
        
        #Calculate results
        acc = np.sum(preds == y_true)/dataset_sizes['test']
        if args.classes == 2:
            auc = roc_auc_score(y_true, y_score)
        else: 
            auc = 0
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        print(f'Test Acc: {acc:.4f} AUC: {auc:.4f}')
        results = {'tn' : tn,
                   'fp' : fp,
                   'fn' : fn,
                   'tp' : tp,
                   'acc': float(acc),
                   'auc': auc}
        #print(y_true, preds, results)
        if args.dim == '3d': #If it is 3D, no need to calculate results for patients
            p = 'patient'    #Write the results of patches as patients
            write_results(args, exp_no, fold, tr_acc, tr_auc, results, 'NA', p)
            print(f'{p} results are saved at {args.resultpath}')
            break
        
        else: #If it is 2D write the results of both patches and patients
            write_results(args, exp_no, fold, tr_acc, tr_auc, results, 'NA', p)
            print(f'{p} results are saved at {args.resultpath}')
            
        
        
def eval_triplet(exp_no, fold, args, model, dataloader, device, dataset_sizes, train_val_test, neighbors):
    
    pp_list = ['patches'] if args.slices in ['mnist', 'cifar10'] else ['patches', 'patients']
    
    #Create empty tensor to store features and labels
    features = {x: torch.FloatTensor().to(device) for x in train_val_test}
    labels = {x: torch.FloatTensor().to(device) for x in train_val_test}
    pids = {x: [] for x in train_val_test}
    binary_labels = {x: torch.FloatTensor().to(device) for x in train_val_test}
    
    #Extract and store features 
    for phase in train_val_test:
        for data in dataloader[phase]:
            if args.slices in ['mnist', 'cifar10']:
                inputs, label, pid = data[0].to(device), data[1].to(device), data[1]
            else:
                inputs, label, pid = data[0].to(device), data[1].to(device), data[3]
            if args.slices == 'mnist':
                label = torch.where(label<5, 0, 1)
            model.train(False)
            with torch.no_grad():
                outputs = model(inputs)
            features[phase] = torch.cat((features[phase],outputs))
            labels[phase] = torch.cat((labels[phase],label))
            pids[phase].extend(pid)
            
    #Add validation set to the train set if it exist
    if 'val' in train_val_test:
        features['train'] = torch.cat((features['train'],features['val']))
        labels['train'] = torch.cat((labels['train'],labels['val']))
        pids['train'].extend(pids['val'])
        dataset_sizes['train'] += dataset_sizes['val']
    
    #Visualize feature map
    #plot_feature_map(features, labels, exp_no, fold)
    
    #Train and evaluate the kNN with different number of neighbors
    for nn in neighbors:
        knn = KNeighborsClassifier(n_neighbors=nn)
        binary_labels['train'] = labels['train'].detach().cpu().numpy()
        binary_labels['test'] = labels['test'].detach().cpu().numpy()
        if args.classes > 2:
            raise InvalidArgument('Multiclass classification is not possible at the moment.\
                                  Check the binary conversion method: error in threshold calculation.')
            
        #Convert labels to binary if it is indicated as before evaluation
        if args.classes > 2 and args.binary_conversion == 'before':
            th = int(args.classes/2)
            binary_labels['train'] = np.where(binary_labels['train']<th, 0, 1)
            binary_labels['test'] = np.where(binary_labels['test']<th, 0, 1)
            
        #Fit the model and get the predictions
        knn.fit(features['train'].cpu(), binary_labels['train'])
        train_score = knn.predict_proba(features['train'].cpu())
        test_score = knn.predict_proba(features['test'].cpu())
        train_preds = np.argmax(train_score, axis=1)
        test_preds = np.argmax(test_score, axis=1)
        
        #Convert labels to binary if it is indicated as after evaluation
        if args.classes > 2 and args.binary_conversion == 'after':
            th = int(args.classes/2)
            train_preds = np.where(train_preds<th, 0, 1)
            test_preds = np.where(test_preds<th, 0, 1)
            binary_labels['train'] = np.where(binary_labels['train']<th, 0, 1)
            binary_labels['test'] = np.where(binary_labels['test']<th, 0, 1)

            train_score = np.transpose(np.array([np.sum(train_score[:,:th], axis=1),
                                   np.sum(train_score[:,th:], axis=1)]))
            test_score = np.transpose(np.array([np.sum(test_score[:,:th], axis=1),
                                   np.sum(test_score[:,th:], axis=1)]))

        for p in pp_list:
            
            # Calculate patient predictions
            if p == 'patients':
                pts_train_preds, pts_test_preds, pts_train_labels, pts_test_labels = [], [], [], []
                pts_train_score, pts_test_score= [], []
                pids = {x: np.array(pids[x]) for x in train_val_test}
                for pt in np.unique(pids['train']):
                    pt_train_preds = train_preds[pids['train']==pt]
                    pt_train_pred = np.sum(pt_train_preds)/len(pt_train_preds)
                    pts_train_preds.append(np.round(pt_train_pred))
                    pts_train_score.append(pt_train_pred)
                    pt_train_labels = binary_labels['train'][pids['train']==pt]
                    pt_train_label = np.round(np.sum(pt_train_labels)/len(pt_train_labels)) #?

                    pts_train_labels.append(pt_train_label)
                    
                for pt in np.unique(pids['test']):
                    pt_test_preds = test_preds[pids['test']==pt]
                    pt_test_pred = np.sum(pt_test_preds)/len(pt_test_preds)
                    pts_test_preds.append(np.round(pt_test_pred))
                    pts_test_score.append(pt_test_pred)
                    pt_test_labels = binary_labels['test'][pids['test']==pt]
                    pt_test_label = np.round(np.sum(pt_test_labels)/len(pt_test_labels)) #?

                    pts_test_labels.append(pt_test_label)
                    
                train_preds = np.array(pts_train_preds)
                test_preds = np.array(pts_test_preds)
                train_score = np.array(pts_train_score)
                test_score = np.array(pts_test_score)
                binary_labels['train'] = np.array(pts_train_labels)
                binary_labels['test'] = np.array(pts_test_labels)
                #dataset_sizes['train'] = len(binary_labels['train'])
                #dataset_sizes['test'] = len(binary_labels['test'])
                
            elif p == 'patches':
                train_score = train_score[:,1]
                test_score = test_score[:,1]

            #Calculate the metrics
            tr_acc = np.sum(np.where(train_preds == binary_labels['train'],1,0))/len(train_preds)
            tr_auc = roc_auc_score(binary_labels['train'], train_score)
            test_acc = np.sum(np.where(test_preds == binary_labels['test'],1,0))/len(test_preds)
            test_auc = roc_auc_score(binary_labels['test'], test_score)
            tn, fp, fn, tp = confusion_matrix(binary_labels['test'], test_preds).ravel()
         
            #Write the results
            results = {'tn' : tn,
                       'fp' : fp,
                       'fn' : fn,
                       'tp' : tp,
                       'acc': float(test_acc),
                       'auc': test_auc}
            
            if args.dim == '3d': #If it is 3D, no need to calculate results for patients
                p = 'patient'    #Write the results of patches as patients
                write_results(args, exp_no, fold, tr_acc, tr_auc, results, nn, p)
                print(f'{p} results are saved at {args.resultpath}')
                break
            
            write_results(args, exp_no, fold, tr_acc, tr_auc, results, nn, p)
            print(f'{p} results are saved at {args.resultpath}')
            
    print(f'Results are saved at {args.resultpath}')
    
    
    
