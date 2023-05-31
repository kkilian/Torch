import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    """
    LeNet implementation for image classification.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.15)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet_dropout(nn.Module):
    """
    LeNet model with dropout for image classification.
    """
    def __init__(self, num_classes=10, dropout=0.1):
        super().__init__()
        
        self.dropout_ratio = dropout
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.15)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):

        p = self.dropout_ratio
        
        out = F.relu(F.dropout(self.conv1(x), p))
        out = F.max_pool2d(out, 2)
        
        out = F.relu(F.dropout(self.conv2(out), p))
        out = F.max_pool2d(out, 2)
        
        out = out.view(out.size(0), -1)
        
        out = F.relu(F.dropout(self.fc1(out), p))
        out = F.relu(F.dropout(self.fc2(out), p))
        
        out = F.dropout(self.fc3(out), p)
        
        return out

class LeNet_BatchNorm(nn.Module):
    """
    LeNet model with batch normalization for image classification.
    """
    def __init__(self, num_classes=10, affine=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6, affine=affine)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16, affine=affine)
        self.fc1 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, affine=affine)
        self.fc2 = nn.Linear(128, 84)
        self.bn4 = nn.BatchNorm1d(84, affine=affine)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
       
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn3(self.fc1(out)))
        out = F.relu(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        return out
