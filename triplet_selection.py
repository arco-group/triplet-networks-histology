import torch
import torch.nn.functional as F

def tsm0(args, criterion, distances, positive_list, negative_list, outputs, i, out_features, device):
    anchors_list = []
    positives_list = []
    negatives_list = []
    m = args.margin
    anchor = outputs[i]

    for p in positive_list:
        positive = outputs[p]
        dp = distances[p]
        for n in negative_list:
            negative = outputs[n]
            dn = distances[n]
            if (dn > dp) and (dn < dp+m):

                anchors_list.append(anchor)
                positives_list.append(positive)
                negatives_list.append(negative)

    if len(anchors_list) == 0:
        return -1
    anchors = torch.stack(anchors_list)
    positives = torch.stack(positives_list)         
    negatives = torch.stack(negatives_list)    

    loss = criterion(anchors, positives, negatives)
    return loss

def tsm1(args, criterion, positives, negatives, outputs, survivals, i, out_features):
    
    triplets = args.triplets
    if triplets > len(positives):
        triplets = len(positives)
    if triplets > len(negatives):
        triplets = len(negatives)
    if triplets == 0:
       return -1
    anchor = outputs[i].expand(triplets, out_features)
    positive = outputs[positives[:triplets]]
    negative = outputs[negatives[:triplets]]

    
    if args.adaptive:
        a_times = survivals[i].expand(triplets)
        p_times = survivals[positives[:triplets]]
        n_times = survivals[negatives[:triplets]]
        margin = abs(abs(a_times-p_times)-abs(a_times-n_times))/750
        loss = criterion(anchor, positive, negative, margin)
    else:
        loss = criterion(anchor, positive, negative)
    return loss

def tsm2(args, diff_th, criterion, positives, negatives, outputs, i, out_features, phase):
    batch_loss = 0
    diff_offset = 0
    while batch_loss < diff_th:
        triplets = args.triplets
        if (diff_offset+triplets) > len(positives):
            triplets = len(positives)-diff_offset
        if triplets > len(negatives):
            triplets = len(negatives)
        if triplets == 0:
            return -1
        anchor = outputs[i].expand(triplets, out_features)
        positive = outputs[positives[diff_offset:(diff_offset+triplets)]]
        negative = outputs[negatives[:triplets]]
        loss = criterion(anchor, positive, negative)
        batch_loss = loss.item()
        diff_offset += 1
        if phase == 'val':
            break
    return loss

def tsm3(args, diff_th, criterion, positives, negatives, outputs, i, out_features, phase):
    batch_loss = 0
    diff_offset = 0
    while batch_loss < diff_th:
        triplets = args.triplets
        if (diff_offset+triplets) > len(positives):
            triplets = len(positives)-diff_offset
        if (diff_offset+triplets+1) > len(negatives):
            triplets = len(negatives)-diff_offset-1
        if triplets == 0:
            return -1
        anchor = outputs[i].expand(triplets, out_features)
        positive = outputs[positives[diff_offset:(diff_offset+triplets)]]
        negative = outputs[negatives[-(diff_offset+triplets+1):-(diff_offset+1)]]
        loss = criterion(anchor, positive, negative)
        batch_loss = loss.item()
        diff_offset += 1
        if phase == 'val':
            break
    return loss

def tsm4(margin, distances, labels):
    one_hot_labels = F.one_hot(labels.to(torch.int64), num_classes=2).to(torch.float)
    label_matrix = torch.matmul(one_hot_labels, one_hot_labels.transpose(0,1))*2-1
    distance_matrix = torch.mul(distances, label_matrix)
    sums = torch.sum(distance_matrix, dim=0) +  margin
    before_max = torch.concat((torch.zeros_like(sums), sums)).reshape(2,-1)
    loss = (1/len(labels))*torch.sum(torch.max(before_max,dim=-2)[0])

    return loss
