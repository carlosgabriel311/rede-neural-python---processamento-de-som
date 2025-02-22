import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, transforms
from src.training_model import Training
from src.models.mini_vgg import VGG_mini 


# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 64
N_CLASSES = 10
DEVICE = 'cpu'


# define transforms
transforms_ = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])
# Carregar o dataset de imagens
dataset = datasets.ImageFolder(root='../rede_neural_python_processamento_de_som/data/imgs/', transform=transforms_)

# Dividir 80% treino e 20% validação
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

# Criar os DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = VGG_mini(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

neural = Training(model, 'vgg_mini',criterion, optimizer, DEVICE, train_loader, valid_loader)

neural.start_training(10)