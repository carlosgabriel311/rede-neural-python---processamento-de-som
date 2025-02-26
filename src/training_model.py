import numpy as np
from datetime import datetime 
import torch
import matplotlib.pyplot as plt

class Training:
    def __init__(self, model, name_model, criterion, optimizer, device, train_loader, valid_loader) :
        '''
        Initialize training variables
        '''
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.name_model = name_model
        self.sum_accuracy = 0
        self.sum_accuracy_squared = 0
        self.count_accuracy = 0

    def start_training(self, epochs, print_every=1):
        '''
        Function defining the entire training loop
        '''
        
        # set objects for storing metrics
        train_losses = []
        valid_losses = []
        # Train model
        for epoch in range(0, epochs):

            # training
            train_loss = self.train()
            train_losses.append(train_loss)

            # validation
            with torch.no_grad():
                valid_loss = self.validate()
                valid_losses.append(valid_loss)

            if epoch % print_every == (print_every - 1):
                
                train_acc = self.get_accuracy(self.train_loader)
                valid_acc = self.get_accuracy(self.valid_loader)

                #Realiza as somas das acuracias para utilização
                self.sum_accuracy += valid_acc
                self.sum_accuracy_squared += valid_acc ** 2
                self.count_accuracy += 1

                #calcula a média
                average_valid_accuracy = self.sum_accuracy / self.count_accuracy 

                # Cálculo do desvio padrão amostral
                if self.count_accuracy > 1:
                    variance = ((self.sum_accuracy_squared / self.count_accuracy) - (average_valid_accuracy ** 2)) * (self.count_accuracy / (self.count_accuracy - 1))
                    std_dev = torch.sqrt(variance)
                else:
                    std_dev = 0  # Se só tiver um valor, desvio padrão é 0

                # Exibir os valores
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                    f'Epoch:{epoch}\t'
                    f'Train loss: {train_loss:.4f}\t'
                    f'Valid loss: {valid_loss:.4f}\t'
                    f'Train accuracy: {100 * train_acc:.2f}\t'
                    f'Valid accuracy: {100 * valid_acc:.2f}\t'
                    f'Valid average accuracy: {100 * average_valid_accuracy:.2f}\t'
                    f'Valid Standard deviation: {100 * std_dev:.2f}')
                
                self.save_to_file(self.name_model, epoch, train_loss, valid_loss, train_acc, valid_acc, average_valid_accuracy, std_dev)

        self.plot_losses(train_losses, valid_losses)
    
    def train(self):
        '''
        Function for the training step of the training loop
        '''
        self.model.train()
        running_loss = 0
        
        
        for X, y_true in self.train_loader:

            self.optimizer.zero_grad()
            
            X = X.to(self.device)
            y_true = y_true.to(self.device)
        
            # Forward pass
            y_hat, _ = self.model(X) 
            loss = self.criterion(y_hat, y_true) 
            running_loss += loss.item() * X.size(0)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def validate(self):
        '''
        Function for the validation step of the training loop
        '''
    
        self.model.eval()
        running_loss = 0
        
        for X, y_true in self.valid_loader:
        
            X = X.to(self.device)
            y_true = y_true.to(self.device)

            # Forward pass and record loss
            y_hat, _ = self.model(X) 
            loss = self.criterion(y_hat, y_true) 
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(self.valid_loader.dataset)
            
        return epoch_loss
    
    def get_accuracy(self, data_loader):
        '''
        Function for computing the accuracy of the predictions over the entire data_loader
        '''
        
        correct_pred = 0 
        n = 0
        
        with torch.no_grad():
            self.model.eval()
            for X, y_true in data_loader:

                X = X.to(self.device)
                y_true = y_true.to(self.device)

                _, y_prob = self.model(X)
                _, predicted_labels = torch.max(y_prob, 1)

                n += y_true.size(0)
                correct_pred += (predicted_labels == y_true).sum()

        return correct_pred.float() / n

    def plot_losses(self, train_losses, valid_losses):
        '''
        Function for plotting training and validation losses
        '''
        
        # temporarily change the style of the plots to seaborn 
        plt.style.use('seaborn-v0_8')

        train_losses = np.array(train_losses) 
        valid_losses = np.array(valid_losses)

        fig, ax = plt.subplots(figsize = (8, 4.5))

        ax.plot(train_losses, color='blue', label='Training loss') 
        ax.plot(valid_losses, color='red', label='Validation loss')
        ax.set(title="Loss over epochs", 
                xlabel='Epoch',
                ylabel='Loss') 
        ax.legend()
        fig.savefig(f'../rede_neural_python_processamento_de_som/data/graphics/training_loss_{self.name_model}', dpi=300)
        
        # change the plot style to default
        plt.style.use('default')

    def save_to_file(self, filename, epoch, train_loss, valid_loss, train_acc, valid_acc, average_valid_accuracy, std_dev):
        # Abre o arquivo no modo de append (adiciona ao final do arquivo)
        with open(filename, 'a') as file:
        # Formata a string que será salva no arquivo
            log_message = (f'{datetime.now().time().replace(microsecond=0)} --- '
                            f'Epoch:{epoch}\t'
                            f'Train loss: {train_loss:.4f}\t'
                            f'Valid loss: {valid_loss:.4f}\t'
                            f'Train accuracy: {100 * train_acc:.2f}\t'
                            f'Valid accuracy: {100 * valid_acc:.2f}\t'
                            f'Valid average accuracy: {100 * average_valid_accuracy:.2f}\t'
                            f'Valid Standard deviation: {100 * std_dev:.2f}\n')
                
                # Escreve a mensagem no arquivo
            file.write(log_message)