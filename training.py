import torch
import numpy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

def train(model: torch.nn.Module, input_features: numpy.ndarray, target_labels: numpy.ndarray, num_labels: int, epochs=1, batch_size=4, optimizer="adamw") -> numpy.ndarray:
    """Trains the model on the given dataset for a number of epochs and returns logged training metrics.

    Args:
        model: The model to be trained. The model should be compatible with the input_feature's shape and produce num_labels outputs for each input.
        input_features: The inputs to the model. Should be an array of type np.float32 with shape (n, ...), where n is the number of datapoints in the dataset. The following dimensions are arbitrary and must be compatible with the model's expeted inputs.
        target_labels: The targets for each sample. Should be an array of type numpy.long with shape (n,), where n is the number of datapoints in the dataset.
        num_labels: The number of different classes in the dataset.
        epochs: The number of epochs for which training should run. (An epoch is a single run through the dataset).
        batch_size: The number of samples that the model processes per training step. The number of steps per epoch is n // batch_size, where n is the total number of samples in the dataset.
        optimizer: Which optimizer to use.
    """
     # Convert input data to PyTorch tensors
    inputs_tensor = torch.tensor(input_features, dtype=torch.float32)
    labels_tensor = torch.tensor(target_labels, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Configure optimizer
    if optimizer == "adamw":
        optimizer = AdamW(model.parameters())
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Define loss function
    loss_fn = CrossEntropyLoss()

    # Track metrics
    metrics = []

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Compute epoch metrics
        accuracy = correct / total
        average_loss = total_loss / len(dataloader)
        metrics.append({"epoch": epoch + 1, "loss": average_loss, "accuracy": accuracy})
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {average_loss:.4f}, Accuracy = {accuracy:.4f}")

    return np.array(metrics)