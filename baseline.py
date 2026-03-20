import numpy as np
import torch
from torch.utils.data import Subset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from dataset import load_gtzan_metadata, GTZANDataset, get_transform


def extract_features(dataset):
    X = []
    y = []

    for i in range(len(dataset)):
        x, label = dataset[i]

        # Flatten spectrogram
        x = x.numpy().flatten()

        X.append(x)
        y.append(label)

    return np.array(X), np.array(y)


def main():
    print("Loading data...")

    file_paths, labels, classes = load_gtzan_metadata()
    transform = get_transform()

    dataset = GTZANDataset(file_paths, labels, transform)

    # Load same split as CNN
    checkpoint = torch.load("best_model.pth", map_location="cpu")
    train_idx = checkpoint["train_indices"]
    test_idx = checkpoint["test_indices"]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    print("Extracting features...")

    X_train, y_train = extract_features(train_dataset)
    X_test, y_test = extract_features(test_dataset)

    print("Training Logistic Regression...")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\nBaseline Test Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=classes))


if __name__ == "__main__":
    main()