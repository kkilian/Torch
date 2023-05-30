import torch
import torch.nn as nn
import numpy as np

def train_model(model, X_train, Y_train, num_epochs=200, batch_size=128, learning_rate=0.001):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    losses, accuracies = [], []
    
    for epoch in range(num_epochs):
        # Randomly sample a batch of training data
        indices = np.random.randint(0, X_train.shape[0], size=(batch_size))
        X = torch.tensor(X_train[indices].reshape((batch_size, 1, 28, 28))).float()
        Y = torch.tensor(Y_train[indices]).long()
        
        model.zero_grad()
        out = model(X)
        cat = torch.argmax(out, dim=1)
        
        accuracy = (cat == Y).float().mean()
        loss = loss_function(out, Y)
        loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss, accuracy = loss.item(), accuracy.item()
        losses.append(loss)
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    
    return losses, accuracies