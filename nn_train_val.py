import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training loop
def nn_train(model, criterion, optimizer, trainloader):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    total_samples = 0
    model.to(device)
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.unsqueeze(1).to(device)
        optimizer.zero_grad()  # Clear the gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update statistics
        train_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    train_loss /= total_samples
    return train_loss


# Validation loop
def nn_validate(model, criterion, valloader):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total_samples = 0
    model.to(device)

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            predicted = torch.round(outputs)  # Apply threshold 0.5 for binary classification
            correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)

    val_loss /= total_samples
    val_accuracy = correct / total_samples
    model.to("cpu")
    return val_loss, val_accuracy