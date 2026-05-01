import sys
import csv
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.fc_model import FCNet
from models.cnn_model import CNNNet
from perturbation.corruptions import (
    add_gaussian_noise,
    rotate_images,
    apply_occlusion,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

RESULTS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_FILE = RESULTS_DIR / "results.csv"


def get_testloader():
    transform = transforms.ToTensor()

    testset = torchvision.datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=False,
        download=True,
        transform=transform,
    )

    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return testloader


def load_model(model_type):
    if model_type == "fc":
        model = FCNet()
        path = PROJECT_ROOT / "models" / "fc_model.pth"
    elif model_type == "cnn":
        model = CNNNet()
        path = PROJECT_ROOT / "models" / "cnn_model.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


def apply_perturbation(images, perturbation, severity):
    if perturbation == "clean":
        return images

    if perturbation == "gaussian_noise":
        return add_gaussian_noise(images, float(severity))

    if perturbation == "rotation":
        return rotate_images(images, float(severity))

    if perturbation == "occlusion":
        return apply_occlusion(images, int(severity))

    raise ValueError(f"Unknown perturbation: {perturbation}")


def evaluate(model, testloader, perturbation, severity):
    correct = 0
    total = 0
    confidence_sum = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            images = apply_perturbation(images, perturbation, severity)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            confidence_sum += confidence.sum().item()

    accuracy = correct / total
    avg_confidence = confidence_sum / total

    return accuracy, avg_confidence


def write_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "perturbation",
                "severity",
                "accuracy",
                "confidence",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {RESULTS_FILE}")


def main():
    testloader = get_testloader()

    experiments = [
        ("clean", 0),

        ("gaussian_noise", 0.10),
        ("gaussian_noise", 0.20),
        ("gaussian_noise", 0.30),
        ("gaussian_noise", 0.40),

        ("rotation", 10),
        ("rotation", 20),
        ("rotation", 30),
        ("rotation", 45),

        ("occlusion", 6),
        ("occlusion", 10),
        ("occlusion", 14),
    ]

    rows = []

    for model_type in ["fc", "cnn"]:
        print(f"\nLoading {model_type} model...")
        model = load_model(model_type)

        for perturbation, severity in experiments:
            print(f"Evaluating {model_type}: {perturbation}, severity={severity}")

            accuracy, confidence = evaluate(
                model,
                testloader,
                perturbation,
                severity,
            )

            rows.append({
                "model": model_type,
                "perturbation": perturbation,
                "severity": severity,
                "accuracy": round(accuracy, 4),
                "confidence": round(confidence, 4),
            })

            print(
                f"  accuracy={accuracy:.4f}, "
                f"confidence={confidence:.4f}"
            )

    write_results(rows)


if __name__ == "__main__":
    main()
