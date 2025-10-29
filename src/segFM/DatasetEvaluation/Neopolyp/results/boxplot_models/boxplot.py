import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# --- Parameters ---
script_dir = os.path.dirname(os.path.abspath(__file__))
METRIC = "DSC"
csv_file = os.path.join(script_dir, "neopolyp.csv")

# --- Load CSV ---
df = pd.read_csv(csv_file)


# --- Boxplot ---
plt.figure(figsize=(8, 4))
sns.boxplot(
    data=df,
    x="Model",
    y=METRIC,
    hue="Mode",
    showfliers=False  # hides outliers for clarity
)

# --- Formatting ---
plt.xlabel("")
plt.ylabel(METRIC, fontsize=14)
plt.legend(title="Prompt Type", loc="lower left")

# Larger ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()


# Save as SVG, PNG and PDF
plt.savefig(os.path.join(script_dir, "boxplot_neopolyp.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, "boxplot_neopolyp.png"), dpi=300)
plt.savefig(os.path.join(script_dir, "boxplot_neopolyp.pdf"), dpi=300)

plt.show()
