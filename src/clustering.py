"""
Customer Segmentation Using Data Science
----------------------------------------
K-Means clustering module.

Responsibilities:
    - Scale clustering features
    - Calculate Elbow Method
    - Train K-Means model
    - Assign cluster labels
    - Calculate cluster centers
    - Calculate Silhouette Score
    - Generate cluster summary
    - Save clustering outputs
"""

import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_customers.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

SEGMENTED_DATA_PATH = os.path.join(
    OUTPUT_DIR,
    "customer_segments.csv"
)

CLUSTER_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "cluster_summary.csv"
)

CLUSTER_CENTERS_PATH = os.path.join(
    OUTPUT_DIR,
    "cluster_centers.csv"
)


# ---------------------------------------------------------
# Load Processed Dataset
# ---------------------------------------------------------

def load_processed_data(file_path=PROCESSED_DATA_PATH):
    """
    Load the cleaned customer dataset.

    Parameters
    ----------
    file_path : str
        Path to cleaned CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned customer dataset.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Processed dataset not found at:\n{file_path}\n\n"
            "Run data_preprocessing.py first."
        )

    df = pd.read_csv(file_path)

    print("=" * 60)
    print("PROCESSED DATASET LOADED")
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ---------------------------------------------------------
# Prepare Features
# ---------------------------------------------------------

def prepare_features(df):
    """
    Select the features used for clustering.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Selected clustering features.
    """

    feature_columns = [
        "annual_income_k$",
        "spending_score_1_100"
    ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing clustering features: {missing_columns}"
        )

    X = df[feature_columns].copy()

    return X


# ---------------------------------------------------------
# Scale Features
# ---------------------------------------------------------

def scale_features(X):
    """
    Standardize clustering features using StandardScaler.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    tuple
        Scaler object and scaled feature array.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\n" + "=" * 60)
    print("FEATURE SCALING COMPLETED")
    print("=" * 60)

    return scaler, X_scaled


# ---------------------------------------------------------
# Elbow Method
# ---------------------------------------------------------

def calculate_elbow(
    X_scaled,
    min_clusters=1,
    max_clusters=10
):
    """
    Calculate K-Means inertia for different K values.

    Parameters
    ----------
    X_scaled : array-like
        Scaled feature data.

    min_clusters : int
        Minimum number of clusters.

    max_clusters : int
        Maximum number of clusters.

    Returns
    -------
    tuple
        Cluster values and inertia values.
    """

    k_values = range(
        min_clusters,
        max_clusters + 1
    )

    inertia = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertia.append(
            model.inertia_
        )

    print("\n" + "=" * 60)
    print("ELBOW METHOD CALCULATION COMPLETED")
    print("=" * 60)

    elbow_results = pd.DataFrame({
        "number_of_clusters": list(k_values),
        "inertia": inertia
    })

    print(elbow_results.to_string(index=False))

    return list(k_values), inertia


# ---------------------------------------------------------
# Train K-Means Model
# ---------------------------------------------------------

def train_kmeans(
    X_scaled,
    n_clusters=5
):
    """
    Train the final K-Means clustering model.

    Parameters
    ----------
    X_scaled : array-like
        Scaled feature data.

    n_clusters : int
        Number of clusters.

    Returns
    -------
    sklearn.cluster.KMeans
        Trained K-Means model.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    print("\n" + "=" * 60)
    print("K-MEANS MODEL TRAINED")
    print("=" * 60)
    print(f"Number of clusters: {n_clusters}")
    print(f"Inertia           : {model.inertia_:.4f}")

    return model


# ---------------------------------------------------------
# Assign Cluster Labels
# ---------------------------------------------------------

def assign_clusters(
    df,
    model,
    X_scaled
):
    """
    Assign K-Means cluster labels to customers.

    Parameters
    ----------
    df : pandas.DataFrame
    model : trained KMeans model
    X_scaled : scaled feature data

    Returns
    -------
    pandas.DataFrame
        Dataset with cluster labels.
    """

    df_clustered = df.copy()

    labels = model.predict(X_scaled)

    df_clustered["cluster"] = labels

    print("\n" + "=" * 60)
    print("CLUSTER LABELS ASSIGNED")
    print("=" * 60)

    print(
        df_clustered["cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return df_clustered


# ---------------------------------------------------------
# Calculate Cluster Centers
# ---------------------------------------------------------

def calculate_cluster_centers(
    model,
    scaler,
    feature_columns
):
    """
    Convert cluster centers back to the original feature scale.

    Parameters
    ----------
    model : trained KMeans model
    scaler : fitted StandardScaler
    feature_columns : list

    Returns
    -------
    pandas.DataFrame
        Cluster centers in original units.
    """

    centers_scaled = model.cluster_centers_

    centers_original = scaler.inverse_transform(
        centers_scaled
    )

    cluster_centers = pd.DataFrame(
        centers_original,
        columns=feature_columns
    )

    cluster_centers.index.name = "cluster"

    cluster_centers = cluster_centers.round(2)

    print("\n" + "=" * 60)
    print("CLUSTER CENTERS")
    print("=" * 60)

    print(cluster_centers.to_string())

    return cluster_centers


# ---------------------------------------------------------
# Calculate Silhouette Score
# ---------------------------------------------------------

def calculate_silhouette(
    X_scaled,
    labels
):
    """
    Calculate the Silhouette Score.

    Parameters
    ----------
    X_scaled : array-like
        Scaled feature data.

    labels : array-like
        Cluster labels.

    Returns
    -------
    float
        Silhouette Score.
    """

    score = silhouette_score(
        X_scaled,
        labels
    )

    print("\n" + "=" * 60)
    print("CLUSTER QUALITY")
    print("=" * 60)
    print(f"Silhouette Score: {score:.4f}")

    return score


# ---------------------------------------------------------
# Generate Cluster Summary
# ---------------------------------------------------------

def generate_cluster_summary(
    df_clustered
):
    """
    Generate descriptive statistics for each cluster.

    Parameters
    ----------
    df_clustered : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Cluster-level summary.
    """

    summary = (
        df_clustered
        .groupby("cluster")
        .agg(
            customer_count=(
                "customerid",
                "count"
            ),

            average_age=(
                "age",
                "mean"
            ),

            average_income=(
                "annual_income_k$",
                "mean"
            ),

            average_spending_score=(
                "spending_score_1_100",
                "mean"
            )
        )
        .round(2)
    )

    summary["customer_percentage"] = (
        summary["customer_count"]
        / len(df_clustered)
        * 100
    ).round(2)

    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)

    print(summary.to_string())

    return summary


# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------

def save_clustering_results(
    df_clustered,
    cluster_summary,
    cluster_centers
):
    """
    Save clustering outputs to the outputs directory.

    Parameters
    ----------
    df_clustered : pandas.DataFrame
        Customer dataset with cluster labels.

    cluster_summary : pandas.DataFrame
        Cluster statistics.

    cluster_centers : pandas.DataFrame
        Cluster centers.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df_clustered.to_csv(
        SEGMENTED_DATA_PATH,
        index=False
    )

    cluster_summary.to_csv(
        CLUSTER_SUMMARY_PATH
    )

    cluster_centers.to_csv(
        CLUSTER_CENTERS_PATH
    )

    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS SAVED")
    print("=" * 60)

    print(
        f"Customer segments : {SEGMENTED_DATA_PATH}"
    )

    print(
        f"Cluster summary   : {CLUSTER_SUMMARY_PATH}"
    )

    print(
        f"Cluster centers   : {CLUSTER_CENTERS_PATH}"
    )


# ---------------------------------------------------------
# Complete Clustering Pipeline
# ---------------------------------------------------------

def run_clustering(
    n_clusters=5
):
    """
    Execute the complete K-Means clustering pipeline.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.

    Returns
    -------
    tuple
        Clustered dataset, cluster summary,
        cluster centers, silhouette score,
        elbow results.
    """

    # Load processed data
    df = load_processed_data()

    # Select features
    X = prepare_features(df)

    # Scale features
    scaler, X_scaled = scale_features(X)

    # Elbow Method
    k_values, inertia = calculate_elbow(
        X_scaled
    )

    # Train final model
    model = train_kmeans(
        X_scaled,
        n_clusters=n_clusters
    )

    # Assign cluster labels
    df_clustered = assign_clusters(
        df,
        model,
        X_scaled
    )

    # Calculate centers
    cluster_centers = calculate_cluster_centers(
        model,
        scaler,
        X.columns.tolist()
    )

    # Silhouette Score
    silhouette = calculate_silhouette(
        X_scaled,
        df_clustered["cluster"]
    )

    # Cluster summary
    cluster_summary = generate_cluster_summary(
        df_clustered
    )

    # Save results
    save_clustering_results(
        df_clustered,
        cluster_summary,
        cluster_centers
    )

    elbow_results = pd.DataFrame({
        "number_of_clusters": k_values,
        "inertia": inertia
    })

    return (
        df_clustered,
        cluster_summary,
        cluster_centers,
        silhouette,
        elbow_results
    )


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":

    try:

        results = run_clustering(
            n_clusters=5
        )

        (
            clustered_data,
            cluster_summary,
            cluster_centers,
            silhouette_score_value,
            elbow_results
        ) = results

        print("\n" + "=" * 60)
        print(
            "CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY"
        )
        print("=" * 60)

        print("\nFinal Customer Segments:")
        print(
            clustered_data[
                [
                    "customerid",
                    "annual_income_k$",
                    "spending_score_1_100",
                    "cluster"
                ]
            ].head(20).to_string(index=False)
        )

        print(
            f"\nFinal Silhouette Score: "
            f"{silhouette_score_value:.4f}"
        )

    except Exception as error:

        print("\n" + "=" * 60)
        print("ERROR DURING CLUSTERING")
        print("=" * 60)
        print(error)