import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        
        # Camadas fully connected
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Funções de ativação
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Achata a entrada (se necessário)
        x = x.view(x.size(0), -1)
        
        # Passa pelas camadas
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Logits (saída da última camada)
        logits = self.fc3(x)
        
        # Probabilidades (softmax aplicado aos logits)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs