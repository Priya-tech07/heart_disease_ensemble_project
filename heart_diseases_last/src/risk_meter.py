import matplotlib.pyplot as plt
import os

def generate_risk_meter(risk_score, save_path="reports/risk_meter.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(6, 2))
    ax = fig.add_subplot(111)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    # Background bands
    ax.axvspan(0, 30, alpha=0.2)
    ax.axvspan(30, 70, alpha=0.2)
    ax.axvspan(70, 100, alpha=0.2)

    # Marker line
    ax.axvline(risk_score, linewidth=3)

    # Labels
    ax.text(10, 0.7, "LOW", fontsize=10, fontweight="bold")
    ax.text(45, 0.7, "MEDIUM", fontsize=10, fontweight="bold")
    ax.text(82, 0.7, "HIGH", fontsize=10, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel(f"Risk Score: {risk_score:.2f} / 100")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path
