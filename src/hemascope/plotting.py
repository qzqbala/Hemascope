"""Единый стиль графиков."""
import matplotlib.pyplot as plt

# WBC, RBC, Platelets — фиолетовый (окраска лейкоцитов), красный, бирюзовый
COLORS = ["#6B4C9A", "#C8423B", "#1F8A8A", "#C9A227", "#4A5568"]


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25,
    })
