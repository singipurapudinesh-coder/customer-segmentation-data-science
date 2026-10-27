"""
Customer Segmentation Using Data Science
----------------------------------------
Main application entry point.

Pipeline:
    1. Data preprocessing
    2. Feature preparation
    3. K-Means clustering
    4. Cluster analysis
    5. Visualization
    6. Final output generation
"""

import os
import sys
import traceback


# ---------------------------------------------------------
# Project Root Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# ---------------------------------------------------------
# Import Project Modules
# ---------------------------------------------------------

from data_preprocessing import preprocess_data
from clustering import run_clustering
from visualization import generate_all_visualizations


# ---------------------------------------------------------
# Display Header
# ---------------------------------------------------------

def display_header():
    """
    Display the project title.
    """

    print("\n")
    print("=" * 75)
    print(
        "             CUSTOMER SEGMENTATION USING DATA SCIENCE"
    )
    print("=" * 75)
    print(
        "              K-Means Clustering Project"
    )
    print("=" * 75)
    print()


# ---------------------------------------------------------
# Display Project Information
# ---------------------------------------------------------

def display_project_information():
    """
    Display project information.
    """

    print("PROJECT INFORMATION")
    print("-" * 75)

    print("Objective:")
    print(
        "Group customers into meaningful segments "
        "using K-Means clustering."
    )

    print("\nTechnologies:")
    print(
        "Python | Pandas | NumPy | Scikit-learn "
        "| Matplotlib | Seaborn"
    )

    print("\nClustering Features:")
    print(
        "Annual Income (k$)"
    )

    print(
        "Spending Score (1-100)"
    )

    print()


# ---------------------------------------------------------
# Run Preprocessing
# ---------------------------------------------------------

def run_preprocessing_stage():
    """
    Execute the data preprocessing stage.

    Returns
    -------
    tuple
        Cleaned dataset and clustering features.
    """

    print("\n")
    print("=" * 75)
    print(
        "[1/3] STARTING DATA PREPROCESSING"
    )
    print("=" * 75)

    cleaned_data, clustering_features = (
        preprocess_data()
    )

    print("\nData preprocessing completed successfully.")

    print(
        f"Cleaned dataset shape: "
        f"{cleaned_data.shape}"
    )

    print(
        f"Clustering feature shape: "
        f"{clustering_features.shape}"
    )

    return (
        cleaned_data,
        clustering_features
    )


# ---------------------------------------------------------
# Run Clustering
# ---------------------------------------------------------

def run_clustering_stage():
    """
    Execute the K-Means clustering stage.

    Returns
    -------
    tuple
        Clustering results.
    """

    print("\n")
    print("=" * 75)
    print(
        "[2/3] STARTING K-MEANS CLUSTERING"
    )
    print("=" * 75)

    results = run_clustering(
        n_clusters=5
    )

    (
        clustered_data,
        cluster_summary,
        cluster_centers,
        silhouette_score,
        elbow_results
    ) = results

    print("\nClustering completed successfully.")

    print(
        f"Number of clusters: "
        f"{len(cluster_centers)}"
    )

    print(
        f"Silhouette Score: "
        f"{silhouette_score:.4f}"
    )

    return results


# ---------------------------------------------------------
# Run Visualization
# ---------------------------------------------------------

def run_visualization_stage():
    """
    Execute the visualization stage.
    """

    print("\n")
    print("=" * 75)
    print(
        "[3/3] STARTING VISUALIZATION"
    )
    print("=" * 75)

    generate_all_visualizations()

    print("\nVisualization stage completed successfully.")


# ---------------------------------------------------------
# Display Final Results
# ---------------------------------------------------------

def display_final_results(
    clustering_results
):
    """
    Display final customer segmentation results.

    Parameters
    ----------
    clustering_results : tuple
        Results returned by run_clustering().
    """

    (
        clustered_data,
        cluster_summary,
        cluster_centers,
        silhouette_score,
        elbow_results
    ) = clustering_results

    print("\n")
    print("=" * 75)
    print(
        "                    FINAL RESULTS"
    )
    print("=" * 75)

    print(
        f"\nTotal Customers : "
        f"{len(clustered_data)}"
    )

    print(
        f"Total Clusters  : "
        f"{len(cluster_centers)}"
    )

    print(
        f"Silhouette Score: "
        f"{silhouette_score:.4f}"
    )

    print("\nCluster Distribution:")
    print("-" * 75)

    cluster_distribution = (
        clustered_data[
            "cluster"
        ]
        .value_counts()
        .sort_index()
    )

    for cluster, count in (
        cluster_distribution.items()
    ):

        percentage = (
            count
            / len(clustered_data)
            * 100
        )

        print(
            f"Cluster {cluster}: "
            f"{count} customers "
            f"({percentage:.2f}%)"
        )

    print("\nCluster Profile:")
    print("-" * 75)

    print(
        cluster_summary.to_string()
    )


# ---------------------------------------------------------
# Display Output Files
# ---------------------------------------------------------

def display_output_files():
    """
    Display generated project files.
    """

    output_dir = os.path.join(
        BASE_DIR,
        "outputs"
    )

    processed_dir = os.path.join(
        BASE_DIR,
        "data",
        "processed"
    )

    print("\n")
    print("=" * 75)
    print(
        "                  GENERATED OUTPUTS"
    )
    print("=" * 75)

    print("\nProcessed Data:")

    processed_files = [
        "cleaned_customers.csv"
    ]

    for filename in processed_files:

        file_path = os.path.join(
            processed_dir,
            filename
        )

        status = (
            "✓ Created"
            if os.path.exists(file_path)
            else "✗ Missing"
        )

        print(
            f"{status}: "
            f"data/processed/{filename}"
        )

    print("\nClustering Results:")

    result_files = [
        "customer_segments.csv",
        "cluster_summary.csv",
        "cluster_centers.csv"
    ]

    for filename in result_files:

        file_path = os.path.join(
            output_dir,
            filename
        )

        status = (
            "✓ Created"
            if os.path.exists(file_path)
            else "✗ Missing"
        )

        print(
            f"{status}: "
            f"outputs/{filename}"
        )

    print("\nVisualization Files:")

    visualization_files = [
        "age_distribution.png",
        "income_distribution.png",
        "spending_score_distribution.png",
        "correlation_matrix.png",
        "customer_distribution_before_clustering.png",
        "elbow_curve.png",
        "cluster_visualization.png",
        "cluster_distribution.png",
        "cluster_profile.png"
    ]

    for filename in visualization_files:

        file_path = os.path.join(
            output_dir,
            filename
        )

        status = (
            "✓ Created"
            if os.path.exists(file_path)
            else "✗ Missing"
        )

        print(
            f"{status}: "
            f"outputs/{filename}"
        )


# ---------------------------------------------------------
# Display Completion Message
# ---------------------------------------------------------

def display_completion_message():
    """
    Display final project completion message.
    """

    print("\n")
    print("=" * 75)
    print(
        "          CUSTOMER SEGMENTATION PROJECT COMPLETED"
    )
    print("=" * 75)

    print("\nNext Steps:")
    print(
        "1. Review the generated cluster visualization."
    )

    print(
        "2. Review the cluster summary."
    )

    print(
        "3. Capture the required internship screenshots."
    )

    print(
        "4. Complete the PDF project documentation."
    )

    print(
        "5. Add the project README."
    )

    print(
        "6. Push the completed project to GitHub."
    )

    print("\nThank you for using the Customer Segmentation System.")
    print("=" * 75)
    print()


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------

def main():
    """
    Execute the complete customer segmentation pipeline.
    """

    display_header()

    display_project_information()

    try:

        # -------------------------------------------------
        # Stage 1: Data Preprocessing
        # -------------------------------------------------

        run_preprocessing_stage()

        # -------------------------------------------------
        # Stage 2: K-Means Clustering
        # -------------------------------------------------

        clustering_results = (
            run_clustering_stage()
        )

        # -------------------------------------------------
        # Stage 3: Visualization
        # -------------------------------------------------

        run_visualization_stage()

        # -------------------------------------------------
        # Final Results
        # -------------------------------------------------

        display_final_results(
            clustering_results
        )

        # -------------------------------------------------
        # Generated Files
        # -------------------------------------------------

        display_output_files()

        # -------------------------------------------------
        # Completion Message
        # -------------------------------------------------

        display_completion_message()

        return 0

    except FileNotFoundError as error:

        print("\n")
        print("=" * 75)
        print(
            "FILE NOT FOUND"
        )
        print("=" * 75)

        print(
            f"\n{error}"
        )

        print(
            "\nPlease verify that the dataset and "
            "required project files exist."
        )

        return 1

    except ValueError as error:

        print("\n")
        print("=" * 75)
        print(
            "DATA VALIDATION ERROR"
        )
        print("=" * 75)

        print(
            f"\n{error}"
        )

        return 1

    except Exception as error:

        print("\n")
        print("=" * 75)
        print(
            "UNEXPECTED PROJECT ERROR"
        )
        print("=" * 75)

        print(
            f"\nError: {error}"
        )

        print("\nDetailed traceback:")
        traceback.print_exc()

        return 1


# ---------------------------------------------------------
# Program Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":

    exit_code = main()

    sys.exit(
        exit_code
    )