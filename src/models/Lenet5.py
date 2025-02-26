import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet5(nn.Module):

    def __init__(self, n_classes):
        super(Lenet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) #transforma a matrix x em vetor.
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    

class Lenet5WithDropout(nn.Module):
    def __init__(self, n_classes):
        super(Lenet5WithDropout, self).__init__()
        
        # Feature extractor (igual ao original)
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        # Classifier com Dropout
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout adicionado aqui
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # Transforma a matriz x em vetor
        
        # Classificação
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs