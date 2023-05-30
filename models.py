import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(1, 6, 5)    # First convolutional layer: input channels = 1, output channels = 6, kernel size5x5
        self.conv2 = nn.Conv2d(6, 16, 5)   # Second convolutional layer: input channels = 6, output channels = 16, kernel size = 5x5
        self.fc1 = nn.Linear(256, 128)     # First fully connected layer: input features = 256, output features = 128
        self.fc2 = nn.Linear(128, 84)      # Second fully connected layer: input features = 128, output features = 84
        self.fc3 = nn.Linear(84, num_classes)  # Third fully connected layer (output layer): input features = 84, output features = num_classes
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize weights of linear layers with a normal distribution (mean=0, std=0.15)
            module.weight.data.normal_(mean=0.0, std=0.15)
            if module.bias is not None:
                # Set bias values to zero
                module.bias.data.zero_()
                
    def forward(self, x):
        out = F.relu(self.conv1(x))          # Apply ReLU activation to the output of the first convolutional layer
        out = F.max_pool2d(out, 2)           # Perform max pooling with a kernel size of 2x2
        out = F.relu(self.conv2(out))        # Apply ReLU activation to the output of the second convolutional layer
        out = F.max_pool2d(out, 2)           # Perform max pooling with a kernel size of 2x2
        out = out.view(out.size(0), -1)      # Flatten the tensor for the fully connected layers
        out = F.relu(self.fc1(out))          # Apply ReLU activation to the output of the first fully connected layer
        out = F.relu(self.fc2(out))          # Apply ReLU activation to the output of the second fully connected layer
        out = self.fc3(out)                  # Output layer (no activation function)
        return out

class LeNet_dropout(nn.Module):
    def __init__(self, num_classes=10, dropout=0.1):
        super().__init__()
        
        # Store dropout ratio
        self.dropout_ratio = dropout
        
        # Define the layers
        self.conv1 = nn.Conv2d(1, 6, 5)     # First convolutional layer: input channels = 1, output channels = 6, kernel size = 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)    # Second convolutional layer: input channels = 6, output channels = 16, kernel size = 5x5
        self.fc1 = nn.Linear(256, 128)      # First fully connected layer: input features = 256, output features = 128
        self.fc2 = nn.Linear(128, 84)       # Second fully connected layer: input features = 128, output features = 84
        self.fc3 = nn.Linear(84, num_classes)   # Third fully connected layer (output layer): input features = 84, output features = num_classes
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize weights of linear layers with a normal distribution (mean=0, std=0.15)
            module.weight.data.normal_(mean=0.0, std=0.15)
            if module.bias is not None:
                # Set bias values to zero
                module.bias.data.zero_()
                
    def forward(self, x):
        p = self.dropout_ratio   # Get the dropout ratio
        
        out = F.relu(F.dropout(self.conv1(x), p))    # Apply dropout after the first convolutional layer and ReLU activation
        out = F.max_pool2d(out, 2)                    # Perform max pooling with a kernel size of 2x2
        
        out = F.relu(F.dropout(self.conv2(out), p))   # Apply dropout after the second convolutional layer and ReLU activation
        out = F.max_pool2d(out, 2)                    # Perform max pooling with a kernel size of 2x2
        
        out = out.view(out.size(0), -1)               # Flatten the tensor for the fully connected layers
        
        out = F.relu(F.dropout(self.fc1(out), p))     # Apply dropout after the first fully connected layer and ReLU activation
        out = F.relu(F.dropout(self.fc2(out), p))     # Apply dropout after the second fully connected layer and ReLU activation
        
        out = F.dropout(self.fc3(out), p)             # Apply dropout after the output layer
        
        return out
    
class LeNet_BatchNorm(nn.Module):
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