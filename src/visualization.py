"""
Customer Segmentation Using Data Science
----------------------------------------
Visualization module.

Responsibilities:
    - Generate exploratory data visualizations
    - Generate Elbow Method plot
    - Generate customer cluster visualization
    - Display cluster centers
    - Generate cluster distribution chart
    - Generate correlation heatmap
    - Save all plots into outputs/
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

SEGMENTED_DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "customer_segments.csv"
)

CLUSTER_SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cluster_summary.csv"
)

CLUSTER_CENTERS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cluster_centers.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# ---------------------------------------------------------
# Visualization Configuration
# ---------------------------------------------------------

sns.set_theme(
    style="whitegrid"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

def load_cleaned_data():
    """
    Load cleaned customer data.

    Returns
    -------
    pandas.DataFrame
    """

    if not os.path.exists(
        PROCESSED_DATA_PATH
    ):
        raise FileNotFoundError(
            "Cleaned dataset not found.\n"
            "Run data_preprocessing.py first."
        )

    return pd.read_csv(
        PROCESSED_DATA_PATH
    )


def load_segmented_data():
    """
    Load customer data containing cluster labels.

    Returns
    -------
    pandas.DataFrame
    """

    if not os.path.exists(
        SEGMENTED_DATA_PATH
    ):
        raise FileNotFoundError(
            "Segmented dataset not found.\n"
            "Run clustering.py first."
        )

    return pd.read_csv(
        SEGMENTED_DATA_PATH
    )


# ---------------------------------------------------------
# Save Plot
# ---------------------------------------------------------

def save_plot(
    filename
):
    """
    Save the current Matplotlib figure.

    Parameters
    ----------
    filename : str
        Output filename.
    """

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------
# Customer Age Distribution
# ---------------------------------------------------------

def plot_age_distribution(
    df
):
    """
    Plot customer age distribution.
    """

    plt.figure(
        figsize=(9, 6)
    )

    sns.histplot(
        df["age"],
        bins=15,
        kde=True
    )

    plt.title(
        "Customer Age Distribution",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Age"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.tight_layout()

    save_plot(
        "age_distribution.png"
    )

    plt.show()


# ---------------------------------------------------------
# Income Distribution
# ---------------------------------------------------------

def plot_income_distribution(
    df
):
    """
    Plot annual income distribution.
    """

    plt.figure(
        figsize=(9, 6)
    )

    sns.histplot(
        df["annual_income_k$"],
        bins=15,
        kde=True
    )

    plt.title(
        "Annual Income Distribution",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Annual Income (k$)"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.tight_layout()

    save_plot(
        "income_distribution.png"
    )

    plt.show()


# ---------------------------------------------------------
# Spending Score Distribution
# ---------------------------------------------------------

def plot_spending_distribution(
    df
):
    """
    Plot customer spending score distribution.
    """

    plt.figure(
        figsize=(9, 6)
    )

    sns.histplot(
        df["spending_score_1_100"],
        bins=15,
        kde=True
    )

    plt.title(
        "Spending Score Distribution",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Spending Score (1-100)"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.tight_layout()

    save_plot(
        "spending_score_distribution.png"
    )

    plt.show()


# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------

def plot_correlation_heatmap(
    df
):
    """
    Generate a correlation heatmap
    for numerical customer features.
    """

    numerical_columns = [
        "age",
        "annual_income_k$",
        "spending_score_1_100"
    ]

    correlation = df[
        numerical_columns
    ].corr()

    plt.figure(
        figsize=(9, 7)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        linewidths=0.5
    )

    plt.title(
        "Customer Feature Correlation",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    save_plot(
        "correlation_matrix.png"
    )

    plt.show()


# ---------------------------------------------------------
# Customer Distribution Before Clustering
# ---------------------------------------------------------

def plot_customer_distribution(
    df
):
    """
    Display customer distribution based on
    income and spending score before clustering.
    """

    plt.figure(
        figsize=(10, 7)
    )

    sns.scatterplot(
        data=df,
        x="annual_income_k$",
        y="spending_score_1_100",
        s=100
    )

    plt.title(
        "Customer Distribution Before Clustering",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Annual Income (k$)"
    )

    plt.ylabel(
        "Spending Score (1-100)"
    )

    plt.tight_layout()

    save_plot(
        "customer_distribution_before_clustering.png"
    )

    plt.show()


# ---------------------------------------------------------
# Elbow Method Plot
# ---------------------------------------------------------

def plot_elbow_method(
    elbow_results
):
    """
    Plot the Elbow Method.

    Parameters
    ----------
    elbow_results : pandas.DataFrame
        Must contain:
        number_of_clusters
        inertia
    """

    required_columns = {
        "number_of_clusters",
        "inertia"
    }

    if not required_columns.issubset(
        elbow_results.columns
    ):
        raise ValueError(
            "Elbow results must contain "
            "'number_of_clusters' and 'inertia'."
        )

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        elbow_results[
            "number_of_clusters"
        ],
        elbow_results[
            "inertia"
        ],
        marker="o",
        linewidth=2
    )

    plt.title(
        "Elbow Method for Optimal Number of Clusters",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.xticks(
        elbow_results[
            "number_of_clusters"
        ]
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    save_plot(
        "elbow_curve.png"
    )

    plt.show()


# ---------------------------------------------------------
# Cluster Visualization
# ---------------------------------------------------------

def plot_clusters(
    df,
    cluster_centers
):
    """
    Visualize customer clusters and their centers.

    Parameters
    ----------
    df : pandas.DataFrame
        Segmented customer dataset.

    cluster_centers : pandas.DataFrame
        Cluster centers in original feature scale.
    """

    required_columns = {
        "annual_income_k$",
        "spending_score_1_100",
        "cluster"
    }

    if not required_columns.issubset(
        df.columns
    ):
        raise ValueError(
            "Segmented dataset is missing "
            "required clustering columns."
        )

    plt.figure(
        figsize=(11, 8)
    )

    sns.scatterplot(
        data=df,
        x="annual_income_k$",
        y="spending_score_1_100",
        hue="cluster",
        palette="Set2",
        s=100,
        alpha=0.85
    )

    # Plot cluster centers
    plt.scatter(
        cluster_centers[
            "annual_income_k$"
        ],
        cluster_centers[
            "spending_score_1_100"
        ],
        marker="X",
        s=300,
        color="black",
        edgecolor="white",
        linewidth=1.5,
        label="Cluster Centers"
    )

    plt.title(
        "Customer Segmentation Using K-Means",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel(
        "Annual Income (k$)",
        fontsize=12
    )

    plt.ylabel(
        "Spending Score (1-100)",
        fontsize=12
    )

    plt.legend(
        title="Cluster"
    )

    plt.tight_layout()

    save_plot(
        "cluster_visualization.png"
    )

    plt.show()


# ---------------------------------------------------------
# Cluster Distribution
# ---------------------------------------------------------

def plot_cluster_distribution(
    df
):
    """
    Display number of customers in each cluster.
    """

    cluster_counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "cluster",
        "customer_count"
    ]

    plt.figure(
        figsize=(9, 6)
    )

    sns.barplot(
        data=cluster_counts,
        x="cluster",
        y="customer_count"
    )

    plt.title(
        "Customer Distribution Across Clusters",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Cluster"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.tight_layout()

    save_plot(
        "cluster_distribution.png"
    )

    plt.show()


# ---------------------------------------------------------
# Cluster Profile Visualization
# ---------------------------------------------------------

def plot_cluster_profile(
    cluster_summary
):
    """
    Visualize average income and spending score
    for each customer cluster.
    """

    profile = cluster_summary.reset_index()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # Average income
    sns.barplot(
        data=profile,
        x="cluster",
        y="average_income",
        ax=axes[0]
    )

    axes[0].set_title(
        "Average Annual Income by Cluster",
        fontweight="bold"
    )

    axes[0].set_xlabel(
        "Cluster"
    )

    axes[0].set_ylabel(
        "Average Income (k$)"
    )

    # Average spending
    sns.barplot(
        data=profile,
        x="cluster",
        y="average_spending_score",
        ax=axes[1]
    )

    axes[1].set_title(
        "Average Spending Score by Cluster",
        fontweight="bold"
    )

    axes[1].set_xlabel(
        "Cluster"
    )

    axes[1].set_ylabel(
        "Average Spending Score"
    )

    plt.tight_layout()

    save_plot(
        "cluster_profile.png"
    )

    plt.show()


# ---------------------------------------------------------
# Generate All Visualizations
# ---------------------------------------------------------

def generate_all_visualizations():
    """
    Generate all project visualizations.
    """

    print("\n" + "=" * 60)
    print(
        "STARTING VISUALIZATION PIPELINE"
    )
    print("=" * 60)

    # Load datasets
    cleaned_df = load_cleaned_data()

    segmented_df = load_segmented_data()

    # Load cluster summary
    if not os.path.exists(
        CLUSTER_SUMMARY_PATH
    ):
        raise FileNotFoundError(
            "cluster_summary.csv not found.\n"
            "Run clustering.py first."
        )

    cluster_summary = pd.read_csv(
        CLUSTER_SUMMARY_PATH
    )

    # Load cluster centers
    if not os.path.exists(
        CLUSTER_CENTERS_PATH
    ):
        raise FileNotFoundError(
            "cluster_centers.csv not found.\n"
            "Run clustering.py first."
        )

    cluster_centers = pd.read_csv(
        CLUSTER_CENTERS_PATH,
        index_col=0
    )

    # Generate visualizations
    print("\n[1/8] Age Distribution")
    plot_age_distribution(
        cleaned_df
    )

    print("\n[2/8] Income Distribution")
    plot_income_distribution(
        cleaned_df
    )

    print("\n[3/8] Spending Score Distribution")
    plot_spending_distribution(
        cleaned_df
    )

    print("\n[4/8] Correlation Heatmap")
    plot_correlation_heatmap(
        cleaned_df
    )

    print("\n[5/8] Customer Distribution")
    plot_customer_distribution(
        cleaned_df
    )

    # Recalculate elbow data
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    feature_columns = [
        "annual_income_k$",
        "spending_score_1_100"
    ]

    X = cleaned_df[
        feature_columns
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    inertia = []

    k_values = range(
        1,
        11
    )

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(
            X_scaled
        )

        inertia.append(
            model.inertia_
        )

    elbow_results = pd.DataFrame({
        "number_of_clusters": list(
            k_values
        ),
        "inertia": inertia
    })

    print("\n[6/8] Elbow Method")
    plot_elbow_method(
        elbow_results
    )

    print("\n[7/8] Customer Clusters")
    plot_clusters(
        segmented_df,
        cluster_centers
    )

    print("\n[8/8] Cluster Distribution")
    plot_cluster_distribution(
        segmented_df
    )

    print("\n[9/9] Cluster Profile")
    plot_cluster_profile(
        cluster_summary
    )

    print("\n" + "=" * 60)
    print(
        "ALL VISUALIZATIONS GENERATED SUCCESSFULLY"
    )
    print("=" * 60)


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":

    try:

        generate_all_visualizations()

    except Exception as error:

        print("\n" + "=" * 60)
        print(
            "ERROR DURING VISUALIZATION"
        )
        print("=" * 60)

        print(error)