import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set(style="whitegrid")
EXPORTS_DIR = "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

def plot_vaccine_gaps(df):
    """
    Creates and saves a bar chart showing vaccine type vs. risk level.

    Args:
        df (DataFrame): Patient dataset with 'Vaccine Type' and 'Risk Level' columns.

    Saves:
        PNG chart as 'exports/vaccine_gap_chart.png'
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="Vaccine Type", hue="Risk Level")
    plt.title("Vaccine Type vs Risk Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = os.path.join(EXPORTS_DIR, "vaccine_gap_chart.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved vaccine gap chart to {filename}")

def plot_age_groups(df):
    """
    Categorizes patients into age groups and visualizes risk level by age group.

    Args:
        df (DataFrame): Patient dataset with 'Age' and 'Risk Level' columns.

    Saves:
        PNG chart as 'exports/age_group_chart.png'
    """
    df["Age Group"] = pd.cut(df["Age"], bins=[0, 29, 59, 120], labels=["<30", "30–59", "60+"])
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Age Group", hue="Risk Level")
    plt.title("Age Groups vs Risk Level")
    plt.tight_layout()
    filename = os.path.join(EXPORTS_DIR, "age_group_chart.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved age group chart to {filename}")
