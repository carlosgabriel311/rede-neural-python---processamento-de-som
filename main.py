import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms

from src.models.mlp import MLP
from src.Preprocessing.add_white_noise import AddWhiteNoise
from src.training_model import Training
from src.models.Lenet5 import Lenet5, Lenet5WithDropout


# parameters
RANDOM_SEED = 40
IMG_SIZE = 32
N_CLASSES = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


learning_rate_list = [0.00005, 0.000025, 0.000075]
n_epochs_list = [30, 30, 50]
batch_size_list = [64, 32, 64]

# Transformações para TREINAMENTO (com data augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.RandomResizedCrop((32, 32), scale=(0.9, 1.0)),  # Pequeno zoom aleatório
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Ajuste de brilho/contraste
    transforms.ToTensor(),
    AddWhiteNoise(std=0.01)
])

# Transformações para VALIDAÇÃO (sem data augmentation)
valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  
    transforms.ToTensor()
])

# Criar datasets a partir dos índices e aplicar as transformações corretas
train_dataset = datasets.ImageFolder(root='../rede_neural_python_processamento_de_som/data/train_imgs/', transform=train_transform)
valid_dataset = datasets.ImageFolder(root='../rede_neural_python_processamento_de_som/data/valid_imgs/', transform=valid_transform)

criterion = nn.CrossEntropyLoss()

#Treinando as 3 configurações da rene neural Lenet5
'''for i in range(0, 3):
    batch_size = batch_size_list[i]
    learning_rate = learning_rate_list[i]
    n_epochs = n_epochs_list[i]

    # Criar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    if (i == 3):
        model = Lenet5WithDropout(N_CLASSES).to(DEVICE)
    else:
        model = Lenet5(N_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    neural = Training(model, f'Lenet5_config{i}', criterion, optimizer, DEVICE, train_loader, valid_loader)
    neural.start_training(n_epochs)'''

#Treinando as 3 configurações do MLP
'''input_size = IMG_SIZE * IMG_SIZE * 3  #tamanho total da imagem
hidden_size_list = [32,64,64] # quantidade de neuronios na camada escondida
batch_size = 64
learning_rate_list = [0.00005, 0.00005, 0.0001]
n_epochs = 30
# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
for i in range(2, 3):
    hidden_size = hidden_size_list[i]
    learning_rate = learning_rate_list[i]
    model = MLP(input_size, hidden_size, N_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    neural = Training(model, f'MLP_config{i}.txt', criterion, optimizer, DEVICE, train_loader, valid_loader)
    neural.start_training(n_epochs)'''
