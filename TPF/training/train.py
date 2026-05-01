import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Allows imports from project root when running: python training/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.fc_model import FCNet
from models.cnn_model import CNNNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "logs"


def get_data():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transform_train,
    )

    testset = torchvision.datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=False,
        download=True,
        transform=transform_test,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    return trainloader, testloader


def train_one_model(model, trainloader, model_name):
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining {model_name} on {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Train Acc: {epoch_acc:.4f}"
        )

    return model


def evaluate(model, testloader):
    model.eval()
    model = model.to(DEVICE)

    correct = 0
    total = 0
    confidence_sum = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            confidence_sum += confidence.sum().item()

    accuracy = correct / total
    avg_confidence = confidence_sum / total

    return accuracy, avg_confidence


def save_model(model, filename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trainloader, testloader = get_data()

    experiments = [
        ("fc", FCNet(), "fc_model.pth"),
        ("cnn", CNNNet(), "cnn_model.pth"),
    ]

    for model_name, model, filename in experiments:
        model = train_one_model(model, trainloader, model_name)

        accuracy, confidence = evaluate(model, testloader)

        print(f"\nClean test results for {model_name}:")
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Confidence: {confidence:.4f}")

        save_model(model, filename)


if __name__ == "__main__":
    main()
