import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Parameters ---
script_dir = os.path.dirname(os.path.abspath(__file__))
METRIC = "NSD"
csv_file = os.path.join(script_dir, "endoscapes.csv")

# --- Load CSV ---
df = pd.read_csv(csv_file)

# --- Combine Model and Mode into one column ---
df["Model_Mode"] = df["Model"].astype(str) + " (" + df["Mode"].astype(str) + ")"

# --- Boxplot ---
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df,
    x="Object",
    y=METRIC,
    hue="Model_Mode",
    #showfliers=False  # hides outliers for clarity
)

# --- Formatting ---
plt.xlabel("Object", fontsize=12)
plt.ylabel(METRIC, fontsize=12)
plt.legend(title="Model (Mode)", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()


# Save as SVG, PNG and PDF
plt.savefig(os.path.join(script_dir, "boxplot_endoscapes.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, "boxplot_endoscapes.png"), dpi=300)
plt.savefig(os.path.join(script_dir, "boxplot_endoscapes.pdf"), dpi=300)

plt.show()
