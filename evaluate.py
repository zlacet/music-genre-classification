import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Subset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_gtzan_metadata, GTZANDataset, get_transform
from model import GenreCNN


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load("best_model.pth", map_location=device)
    classes = checkpoint["classes"]
    test_indices = checkpoint["test_indices"]

    file_paths, labels, _ = load_gtzan_metadata()
    transform = get_transform()

    dataset = GTZANDataset(file_paths, labels, transform=transform)
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = GenreCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    preds, true = evaluate_model(model, test_loader, device)

    accuracy = (preds == true).mean()
    print(f"\nBest Model Test Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(true, preds, target_names=classes))

    cm = confusion_matrix(true, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    idx = random.randint(0, len(test_dataset) - 1)
    sample_input, sample_label = test_dataset[idx]
    sample_input = sample_input.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sample_input)
        pred = torch.argmax(output, dim=1).item()

    print("\nRandom Test Sample Prediction:")
    print(f"Actual Genre: {classes[sample_label]}")
    print(f"Predicted Genre: {classes[pred]}")


if __name__ == "__main__":
    main()