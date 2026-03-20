import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import load_gtzan_metadata, GTZANDataset, get_transform
from model import GenreCNN


SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
STEP_SIZE = 5
GAMMA = 0.5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

MODEL_PATH = "best_model.pth"
LOSS_PLOT_PATH = "loss_curve.png"
ACCURACY_PLOT_PATH = "accuracy_curve.png"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_paths, labels, classes = load_gtzan_metadata()
    transform = get_transform()
    dataset = GTZANDataset(file_paths, labels, transform)

    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = int(VAL_RATIO * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = GenreCNN(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                total_loss += loss.item() * x.size(0)
                preds = torch.argmax(out, dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()

        val_loss = total_loss / total
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {time.time()-start_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "train_indices": train_dataset.indices,
                    "val_indices": val_dataset.indices,
                    "test_indices": test_dataset.indices,
                },
                MODEL_PATH,
            )

        scheduler.step()

    # Loss plot
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Loss")
    plt.savefig(LOSS_PLOT_PATH)

    # Accuracy plot
    plt.figure()
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(ACCURACY_PLOT_PATH)

    print(f"Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()