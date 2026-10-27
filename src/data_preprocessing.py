"""
Customer Segmentation Using Data Science
----------------------------------------
Data preprocessing module.

Responsibilities:
    - Load the customer dataset
    - Validate required columns
    - Clean duplicate and missing records
    - Standardize column names
    - Select clustering features
    - Save the cleaned dataset
"""

import os
import pandas as pd


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Mall_Customers.csv"
)

PROCESSED_DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

PROCESSED_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "cleaned_customers.csv"
)


# ---------------------------------------------------------
# Required Columns
# ---------------------------------------------------------

REQUIRED_COLUMNS = {
    "customerid",
    "gender",
    "age",
    "annual_income_k$",
    "spending_score_1_100"
}


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

def load_dataset(file_path=RAW_DATA_PATH):
    """
    Load the customer dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded customer dataset.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Please place Mall_Customers.csv inside data/raw/"
        )

    df = pd.read_csv(file_path)

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ---------------------------------------------------------
# Standardize Column Names
# ---------------------------------------------------------

def standardize_columns(df):
    """
    Standardize column names for easier processing.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "_")
    )

    return df


# ---------------------------------------------------------
# Validate Dataset
# ---------------------------------------------------------

def validate_dataset(df):
    """
    Validate that all required columns are present.

    Parameters
    ----------
    df : pandas.DataFrame

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            f"{sorted(missing_columns)}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    print("Dataset validation successful.")


# ---------------------------------------------------------
# Clean Dataset
# ---------------------------------------------------------

def clean_dataset(df):
    """
    Clean the customer dataset.

    Operations:
        - Remove duplicate records
        - Handle missing values
        - Convert numerical columns to numeric format
        - Remove invalid numerical records

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """

    df = df.copy()

    initial_rows = len(df)

    # Remove duplicate records
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

    # Convert numerical columns
    numerical_columns = [
        "age",
        "annual_income_k$",
        "spending_score_1_100"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows with missing required values
    missing_before = df.isnull().sum().sum()

    df = df.dropna(
        subset=[
            "customerid",
            "gender",
            "age",
            "annual_income_k$",
            "spending_score_1_100"
        ]
    )

    # Remove invalid values
    df = df[
        (df["age"] > 0)
        & (df["annual_income_k$"] > 0)
        & (df["spending_score_1_100"] >= 1)
        & (df["spending_score_1_100"] <= 100)
    ]

    df = df.reset_index(drop=True)

    final_rows = len(df)

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETED")
    print("=" * 60)
    print(f"Initial rows       : {initial_rows}")
    print(f"Duplicates removed : {duplicate_count}")
    print(f"Missing values     : {missing_before}")
    print(f"Final rows         : {final_rows}")
    print(f"Rows removed       : {initial_rows - final_rows}")

    return df


# ---------------------------------------------------------
# Select Clustering Features
# ---------------------------------------------------------

def select_clustering_features(df):
    """
    Select features used by the K-Means clustering model.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Clustering feature dataset.
    """

    feature_columns = [
        "annual_income_k$",
        "spending_score_1_100"
    ]

    X = df[feature_columns].copy()

    print("\n" + "=" * 60)
    print("CLUSTERING FEATURES SELECTED")
    print("=" * 60)
    print(feature_columns)

    return X


# ---------------------------------------------------------
# Save Processed Dataset
# ---------------------------------------------------------

def save_processed_data(
    df,
    output_path=PROCESSED_DATA_PATH
):
    """
    Save cleaned customer data.

    Parameters
    ----------
    df : pandas.DataFrame
    output_path : str
        Destination CSV path.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("PROCESSED DATA SAVED")
    print("=" * 60)
    print(output_path)


# ---------------------------------------------------------
# Complete Preprocessing Pipeline
# ---------------------------------------------------------

def preprocess_data():
    """
    Execute the complete preprocessing pipeline.

    Returns
    -------
    tuple
        cleaned DataFrame and clustering features.
    """

    # Load dataset
    df = load_dataset()

    # Standardize columns
    df = standardize_columns(df)

    # Validate columns
    validate_dataset(df)

    # Clean data
    df_clean = clean_dataset(df)

    # Select clustering features
    X = select_clustering_features(df_clean)

    # Save cleaned dataset
    save_processed_data(df_clean)

    return df_clean, X


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":

    try:
        cleaned_data, clustering_features = preprocess_data()

        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("\nCleaned Dataset Preview:")
        print(cleaned_data.head())

        print("\nClustering Features Preview:")
        print(clustering_features.head())

    except Exception as error:
        print("\n" + "=" * 60)
        print("ERROR DURING DATA PREPROCESSING")
        print("=" * 60)
        print(error)