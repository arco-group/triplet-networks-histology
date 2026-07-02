import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from utils import InvalidArgument

class TripletNet(torch.nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, fs)
        self.dropout1 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.conv2(x), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.conv3(x), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.conv4(x), negative_slope=0.1))
        x = torch.flatten(x, 1)
        x = self.fc(self.dropout1(x))
        return x
    
class CustomNet(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, out_features)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        x = self.pool(self.bn(F.leaky_relu(self.conv1(x), negative_slope=0.1)))
        x = self.pool(self.bn(F.leaky_relu(self.conv2(x), negative_slope=0.1)))
        x = self.pool(self.bn(F.leaky_relu(self.conv3(x), negative_slope=0.1)))
        x = self.pool(self.bn(F.leaky_relu(self.conv4(x), negative_slope=0.1)))
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout1(x))
        x = F.leaky_relu(self.fc2(self.dropout2(x)), negative_slope=0.1)
        return torch.sigmoid(x)
    
class FFN(nn.Module):
    
    def __init__(self, out_features):
        super().__init__()
        self.layer1 = nn.Linear(199,2048)
        self.layer2 = nn.Linear(2048,1024)
        self.layer3 = nn.Linear(1024,512)
        self.fc = nn.Linear(512,out_features)
        self.bn = nn.BatchNorm1d(512)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.bn(F.relu(self.layer3(x)))
        x = F.relu(self.fc(x))
        return x
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)
        
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

def init_model(model_name, device, feature_size):
    """ Initilizes model.
    
    Args:
        model_name: name of the model
        device: device
        feature_size: output feature size
        
    Returns:
        model and output feature size
    """
    
    if model_name == 'resnet3d':
        
        #Load pretrained weights
        if torchvision.__version__ >= '0.13.0':    
            model = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
        else:
            model = models.video.r3d_18(pretrained=True)
            
        #Remove final FC and take the features of the layer before
        if feature_size == -1: 
            model = torch.nn.Sequential(*list(model.children())[:-1])
            out_features = 512
        
        #Change the final FC according to feature size
        else: 
            model.fc = nn.Linear(model.fc.in_features, feature_size)
            out_features = model.fc.out_features
            
    elif model_name == 'mobilenet':
        
        if torchvision.__version__ >= '0.13.0':    
            model = models.mobilenet_v2(weights='DEFAULT')
        else:
            model = models.mobilenet_v2(pretrained=True)
        
        if feature_size == -1: 
            model = torch.nn.Sequential(*list(model.children())[:-1])
            out_features = 1280

        else: 
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, feature_size)
            out_features = model.classifier[1].out_features
      
        
    elif model_name == 'resnet50':
        
        #Load pretrained weights
        if torchvision.__version__ >= '0.13.0':    
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(pretrained=True)
        
        #Remove final FC and take the features of the layer before
        if feature_size == -1: 
            model = torch.nn.Sequential(*list(model.children())[:-1])
            out_features = 2048
        #Change the final FC according to feature size
        else: 
            model.fc = nn.Linear(model.fc.in_features, feature_size)
            out_features = model.fc.out_features
            
            
    elif model_name == 'vgg16':
        #Load pretrained weights
        if torchvision.__version__ >= '0.13.0':    
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        else:
            model = models.vgg16_bn(pretrained=True)
        #Remove final FC and take the features of the layer before
        if feature_size == -1: 
            model = torch.nn.Sequential(*list(model.children())[:-1])
            out_features = 4096
        #Change the final FC according to feature size
        else: 
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, feature_size)
            out_features = model.classifier[6].out_features
            
    elif model_name == 'googlenet':
        #Load pretrained weights
        if torchvision.__version__ >= '0.13.0':    
            model = models.googlenet(weights='DEFAULT')
        else:
            model = models.googlenet(pretrained=True)
        #Remove final FC and take the features of the layer before
        if feature_size == -1: 
            model = torch.nn.Sequential(*list(model.children())[:-1])
            out_features = 1024
        #Change the final FC according to feature size
        else: 
            model.fc = nn.Linear(model.fc.in_features, feature_size)
            out_features = model.fc.out_features
            
    elif model_name == 'densenet':
        #Load pretrained weights 
        if torchvision.__version__ >= '0.13.0':    
            model = models.densenet161(weights='DEFAULT')
        else:
            model = models.densenet161(pretrained=True)
        #Remove final FC and take the features of the layer before
        if feature_size == -1: 
            model = torch.nn.Sequential(*list(model.children())[:-1])
            out_features = 2208
        #Change the final FC according to feature size
        else: 
            model.classifier = nn.Linear(model.classifier.in_features, feature_size)
            out_features = model.classifier.out_features
          
    elif model_name == 'shufflenet':
        #Load pretrained weights 
        if torchvision.__version__ >= '0.13.0':    
            model = models.shufflenet_v2_x1_0(weights='DEFAULT')
        else:
            model = models.shufflenet_v2_x1_0(pretrained=True)
        #Remove final FC and take the features of the layer before
        if feature_size == -1: 
            model = torch.nn.Sequential(*list(model.children())[:-1])
            out_features = 1024
        #Change the final FC according to feature size
        else: 
            model.fc = nn.Linear(model.fc.in_features, feature_size)
            out_features = model.fc.out_features
            
    
    elif model_name == 'cnn2d':
        model = CustomNet(feature_size)
        out_features = model.fc2.out_features
        model.apply(weights_init)
        
    elif model_name == 'triplet':
        model = TripletNet(feature_size)
        out_features = model.fc.out_features
        model.apply(weights_init)
        
    elif model_name == 'ffn':
        model = FFN(feature_size)
        out_features = model.fc.out_features
        model.apply(weights_init)
        
    else:
        raise InvalidArgument(f'{model_name} is not a valid model name! Please try another.')
        
    return model.to(device), out_features
