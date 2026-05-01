from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "results" / "logs" / "results.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def plot_accuracy(df, perturbation):
    subset = df[df["perturbation"] == perturbation]

    plt.figure()

    for model in subset["model"].unique():
        model_data = subset[subset["model"] == model]
        plt.plot(
            model_data["severity"],
            model_data["accuracy"],
            marker="o",
            label=model,
        )

    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy under {perturbation}")
    plt.legend()
    plt.grid(True)

    output_path = PLOTS_DIR / f"accuracy_{perturbation}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    for perturbation in ["gaussian_noise", "rotation", "occlusion"]:
        plot_accuracy(df, perturbation)


if __name__ == "__main__":
    main()
