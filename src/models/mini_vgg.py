import torch.nn as nn
import torch.nn.functional as F

class VGG_mini(nn.Module):
    
    def __init__(self, n_classes): 
        super(VGG_mini, self).__init__()
        
        # Maxpool 2x2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv layers with batch norm
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding = 1)
        self.norm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(64)
               
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.norm4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.norm5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.norm6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3, padding = 1)
        self.norm7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 512, 3, padding = 1)
        self.norm8 = nn.BatchNorm2d(512)
        
        # fully connected layer with batch norm

        self.fc1 = nn.Linear(512 * 4 * 4, 128)
        self.norm9 = nn.BatchNorm1d(128)
       
        self.fc2 = nn.Linear(128, 64)
        self.norm10 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, n_classes)
        

    def forward(self, x):       
        
        out = F.elu(self.norm1(self.conv1(x)))
        out = F.elu(self.norm2(self.conv2(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm3(self.conv3(out)))
        out = F.elu(self.norm4(self.conv4(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm5(self.conv5(out)))
        out = F.elu(self.norm6(self.conv6(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm7(self.conv7(out)))
        out = F.elu(self.norm8(self.conv8(out)))
        
        out = out.view(-1, 512 * 4 * 4)
        
        out = F.elu(self.norm9(self.fc1(out)))
        out = F.elu(self.norm10(self.fc2(out)))
        out = self.fc3(out)
        probs = F.softmax(out, dim=1)

        return out, probs