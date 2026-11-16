"""
Titanic Dataset - Data Cleansing Pipeline
==========================================

Dataset : Titanic - Machine Learning from Disaster (Kaggle, train.csv)
          https://www.kaggle.com/competitions/titanic/data
Purpose : Load the raw data, find and fix every data-quality problem
          (missing values, hidden missing values, duplicates, inconsistent
          text, wrong data types, outliers) and save a clean dataset.

Run from the project root:
    python src/data_cleansing.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # render charts without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "titanic_raw.csv"
CLEAN_PATH = ROOT / "data" / "cleaned" / "titanic_cleaned.csv"
IMG_DIR = ROOT / "images"

# Chart colours (validated colour-blind-safe palette)
MISSING_COLOR = "#eb6834"   # orange  -> a missing cell
PRESENT_COLOR = "#e9e8e4"   # light grey -> a filled cell
BAR_COLOR = "#2a78d6"       # blue
INK = "#52514e"             # axis / label text


# ----------------------------------------------------------------------------
# STEP 1 - Load the dataset
# ----------------------------------------------------------------------------
def load_data(path=RAW_PATH):
    """Read the raw CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ----------------------------------------------------------------------------
# STEP 2 - Initial inspection
# ----------------------------------------------------------------------------
def inspect_data(df):
    """Print structure, data types and summary statistics."""
    print("First 5 rows:")
    print(df.head().to_string())
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nSummary statistics (numeric columns):")
    print(df.describe().round(2).to_string())


# ----------------------------------------------------------------------------
# STEP 3 - Check for duplicate values
# ----------------------------------------------------------------------------
def check_duplicates(df):
    """Check for fully duplicated rows and duplicated passenger IDs."""
    full_dups = df.duplicated().sum()
    id_dups = df["PassengerId"].duplicated().sum()
    # Same passenger entered twice under a different ID?
    content_dups = df.drop(columns="PassengerId").duplicated().sum()

    print(f"Fully duplicated rows            : {full_dups}")
    print(f"Duplicated PassengerId values    : {id_dups}")
    print(f"Duplicated rows (ignoring the ID): {content_dups}")

    if full_dups == 0 and id_dups == 0 and content_dups == 0:
        print("Result: NO DUPLICATE VALUES FOUND - no rows removed.")
    else:
        df = df.drop_duplicates(subset=df.columns.drop("PassengerId"))
        print(f"Duplicates removed. New shape: {df.shape}")
    return df


# ----------------------------------------------------------------------------
# STEP 4 - Missing-value report
# ----------------------------------------------------------------------------
def missing_value_report(df, label="before"):
    """Count and percentage of missing values in each column."""
    report = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2),
    })
    print(f"Missing values ({label} cleaning):")
    print(report.to_string())
    print(f"Total missing cells: {int(report['Missing Count'].sum())}")
    return report


def plot_missing(df, label):
    """Save a heat-map of missing cells and a bar chart of missing counts."""
    IMG_DIR.mkdir(exist_ok=True)

    # Heat-map: one row per passenger, orange = missing
    fig, ax = plt.subplots(figsize=(10, 4.8))
    cmap = matplotlib.colors.ListedColormap([PRESENT_COLOR, MISSING_COLOR])
    ax.imshow(df.isnull().values.astype(int), aspect="auto",
              cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", color=INK)
    ax.set_ylabel("Row (passenger)", color=INK)
    total = int(df.isnull().sum().sum())
    ax.set_title(f"Missing-value map {label} cleaning  "
                 f"(orange = missing, {total} missing cells)",
                 loc="left", fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"missing_heatmap_{label}.png", dpi=150)
    plt.close(fig)

    # Bar chart of missing counts per column
    counts = df.isnull().sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR, width=0.6)
    for b, v in zip(bars, counts.values):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v),
                    ha="center", fontsize=10, color="#0b0b0b")
    ax.set_ylim(0, max(counts.max() * 1.15, 10))
    ax.set_ylabel("Missing values", color=INK)
    ax.set_title(f"Missing values per column {label} cleaning",
                 loc="left", fontsize=12)
    ax.tick_params(axis="x", rotation=45, colors=INK)
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="y", color="#e9e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"missing_bar_{label}.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# STEP 5 - Clean text columns (whitespace and case)
# ----------------------------------------------------------------------------
def clean_text(df):
    """Trim stray spaces and standardise the case of text columns."""
    df = df.copy()
    text_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
    for col in text_cols:
        before = df[col].copy()
        df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
        changed = (before.fillna("") != df[col].fillna("")).sum()
        print(f"{col:<9}: {changed} value(s) had extra whitespace removed")
    df["Sex"] = df["Sex"].str.lower()
    df["Embarked"] = df["Embarked"].str.upper()
    print("Sex categories     :", sorted(df["Sex"].unique()))
    print("Embarked categories:", sorted(df["Embarked"].dropna().unique()))
    return df


# ----------------------------------------------------------------------------
# STEP 6 - Extract and standardise passenger titles
# ----------------------------------------------------------------------------
def extract_title(df):
    """Pull the title (Mr, Mrs, Miss...) out of Name; group rare titles."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    # Same meaning, different spelling -> one label
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss",
                                       "Mme": "Mrs"})
    # Titles held by very few passengers -> "Rare"
    rare = ["Dr", "Rev", "Major", "Col", "Don", "Lady", "Sir", "Capt",
            "the Countess", "Jonkheer"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    print("Title counts after standardising:")
    print(df["Title"].value_counts().to_string())
    return df


# ----------------------------------------------------------------------------
# STEP 7 - Handle missing values in Embarked (2 missing)
# ----------------------------------------------------------------------------
def impute_embarked(df):
    """Fill the 2 missing ports of embarkation with the mode."""
    df = df.copy()
    mode_port = df["Embarked"].mode()[0]
    missing_rows = df[df["Embarked"].isnull()]["PassengerId"].tolist()
    df["Embarked"] = df["Embarked"].fillna(mode_port)
    print(f"Rows with missing Embarked: PassengerId {missing_rows}")
    print(f"Filled with the most frequent port (mode): '{mode_port}'")
    print(f"Missing Embarked now: {df['Embarked'].isnull().sum()}")
    return df


# ----------------------------------------------------------------------------
# STEP 8 - Handle missing values in Age (177 missing)
# ----------------------------------------------------------------------------
def impute_age(df):
    """Fill missing ages with the median age of the same Title and Pclass."""
    df = df.copy()
    n_missing = df["Age"].isnull().sum()
    group_median = df.groupby(["Title", "Pclass"])["Age"].transform("median")
    medians_used = df.groupby(["Title", "Pclass"], observed=True)["Age"].median()
    df["Age_was_missing"] = df["Age"].isnull().astype(int)   # keep a record
    df["Age"] = df["Age"].fillna(group_median)
    df["Age"] = df["Age"].fillna(df["Age"].median())          # safety net
    print(f"Missing Age values filled: {n_missing}")
    print("Median age used per Title / Pclass group:")
    print(medians_used.unstack().round(1).fillna("-").to_string())
    print(f"Missing Age now: {df['Age'].isnull().sum()}")
    return df


# ----------------------------------------------------------------------------
# STEP 9 - Handle missing values in Cabin (687 missing = 77%)
# ----------------------------------------------------------------------------
def handle_cabin(df):
    """Cabin is 77% empty: keep the information that exists, label the rest."""
    df = df.copy()
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    # Deck = first letter of the cabin code (A-G, T); U = unknown
    df["Deck"] = df["Cabin"].str[0].where(df["Cabin"] != "Unknown", "U")
    print("Cabin missing values replaced with 'Unknown'")
    print("New column HasCabin (1 = cabin recorded):")
    print(df["HasCabin"].value_counts().to_string())
    print("New column Deck:")
    print(df["Deck"].value_counts().to_string())
    return df


# ----------------------------------------------------------------------------
# STEP 10 - Hidden missing values: Fare = 0
# ----------------------------------------------------------------------------
def fix_zero_fares(df):
    """A fare of 0.00 is not a real ticket price - treat it as missing."""
    df = df.copy()
    zero = df["Fare"] == 0
    print(f"Passengers with Fare = 0: {zero.sum()} "
          f"(Pclass counts: {df.loc[zero, 'Pclass'].value_counts().to_dict()})")
    df.loc[zero, "Fare"] = np.nan
    df["Fare"] = df["Fare"].fillna(
        df.groupby("Pclass")["Fare"].transform("median"))
    print("Replaced with the median fare of the passenger's class:")
    print(df.groupby("Pclass")["Fare"].median().round(2).to_string())
    print(f"Fare = 0 now: {(df['Fare'] == 0).sum()}")
    return df


# ----------------------------------------------------------------------------
# STEP 11 - Correct data types
# ----------------------------------------------------------------------------
def convert_dtypes(df):
    """Give every column the data type that matches its meaning."""
    df = df.copy()
    for col in ["Sex", "Embarked", "Title", "Deck"]:
        df[col] = df[col].astype("category")
    df["Pclass"] = pd.Categorical(df["Pclass"], categories=[1, 2, 3],
                                  ordered=True)
    df["Survived"] = df["Survived"].astype("int8")
    df["Age"] = df["Age"].round(1)
    df["Fare"] = df["Fare"].round(2)
    print(df.dtypes.to_string())
    return df


# ----------------------------------------------------------------------------
# STEP 12 - Outlier detection (IQR method)
# ----------------------------------------------------------------------------
def detect_outliers(df):
    """Flag values outside 1.5 x IQR and check they are realistic."""
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        print(f"{col:<6} min {df[col].min():>5}  max {df[col].max():>6} | "
              f"bounds [{low:6.2f}, {high:5.2f}] | outliers: {n_out}")

    # Plot Fare by class to show the outliers are genuine 1st-class fares
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [df.loc[df["Pclass"] == c, "Fare"] for c in [1, 2, 3]]
    ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#cfe1f7", color=BAR_COLOR),
               medianprops=dict(color="#0b0b0b"),
               whiskerprops=dict(color=BAR_COLOR),
               capprops=dict(color=BAR_COLOR),
               flierprops=dict(marker="o", markersize=5,
                               markerfacecolor=MISSING_COLOR,
                               markeredgecolor="white"))
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1st class", "2nd class", "3rd class"], color=INK)
    ax.set_xlabel("Fare (GBP)", color=INK)
    ax.set_title("Fare distribution by passenger class (orange = IQR outlier)",
                 loc="left", fontsize=12)
    ax.grid(axis="x", color="#e9e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fare_outliers_boxplot.png", dpi=150)
    plt.close(fig)
    print("Decision: outliers are KEPT - they are real values")
    print("(large families, luxury 1st-class suites), not entry errors.")


# ----------------------------------------------------------------------------
# STEP 13 - Derived columns
# ----------------------------------------------------------------------------
def engineer_features(df):
    """Add simple, analysis-ready columns built from cleaned data."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 120],
                            labels=["Child", "Teen", "Young Adult",
                                    "Adult", "Senior"])
    print(df[["FamilySize", "IsAlone", "AgeGroup"]].describe(include="all")
            .loc[["count", "unique", "top", "mean", "max"]].to_string())
    return df


# ----------------------------------------------------------------------------
# STEP 14 - Final validation and save
# ----------------------------------------------------------------------------
def validate_and_save(df, path=CLEAN_PATH):
    """Assert the cleaned data meets every quality rule, then save it."""
    assert df.isnull().sum().sum() == 0, "missing values remain"
    assert df.duplicated().sum() == 0, "duplicates remain"
    assert df["Age"].between(0, 100).all(), "invalid age"
    assert (df["Fare"] > 0).all(), "invalid fare"
    assert set(df["Sex"].unique()) == {"male", "female"}
    assert set(df["Embarked"].unique()) <= {"S", "C", "Q"}
    print("All validation checks passed:")
    print("  - 0 missing values      - 0 duplicate rows")
    print("  - Age within 0-100      - every Fare > 0")
    print("  - valid Sex / Embarked categories")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned dataset saved: {path.relative_to(ROOT)}")
    print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ----------------------------------------------------------------------------
# Run the full pipeline
# ----------------------------------------------------------------------------
def main():
    df = load_data()
    inspect_data(df)
    df = check_duplicates(df)
    missing_value_report(df, "before")
    plot_missing(df, "before")
    df = clean_text(df)
    df = extract_title(df)
    df = impute_embarked(df)
    df = impute_age(df)
    df = handle_cabin(df)
    df = fix_zero_fares(df)
    df = convert_dtypes(df)
    detect_outliers(df)
    df = engineer_features(df)
    missing_value_report(df, "after")
    plot_missing(df, "after")
    validate_and_save(df)


if __name__ == "__main__":
    main()
