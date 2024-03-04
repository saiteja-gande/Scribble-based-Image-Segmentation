import torch

# Training loop
def cn_train(model, criterion, optimizer, trainloader):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    total_samples = 0

    for inputs, labels in trainloader:
#         labels = labels.unsqueeze(1)
        optimizer.zero_grad()  # Clear the gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        model.fc2.weight.data.clamp_(0)
        model.fc3.weight.data.clamp_(0)
        
#         print(np.min((model.fc2.weight.data).numpy()))
#         print(np.min((model.fc3.weight.data).numpy()))

        # Update statistics
        train_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    train_loss /= total_samples
    return train_loss


# Validation loop
def cn_validate(model, criterion, valloader):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in valloader:
#           labels = labels.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            predicted = torch.round(outputs)  # Apply threshold 0.5 for binary classification
            correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)

    val_loss /= total_samples
    val_accuracy = correct / total_samples
    return val_loss, val_accuracy