
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

RAW_DATA = DATA_DIR / "raw" / "Mall_Customers.csv"
CLEAN_DATA = DATA_DIR / "processed" / "cleaned_customers.csv"
SEGMENTS_DATA = DATA_DIR / "processed" / "customer_segments.csv"

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📊 Customer Segmentation Using Data Science")
st.subheader("Interactive Customer Analytics Dashboard")

st.markdown(
    """
    Explore customer demographics, spending behavior, and
    K-Means customer segments using this data science project.
    """
)

st.caption(
    "Technology: Python · Pandas · Scikit-learn · K-Means · Streamlit"
)

# --------------------------------------------------
# Load data safely
# --------------------------------------------------
@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return None


segments = load_csv(str(SEGMENTS_DATA))
cleaned = load_csv(str(CLEAN_DATA))
raw = load_csv(str(RAW_DATA))

# Prefer the clustered dataset for customer-level analysis.
customers = segments if segments is not None else cleaned
if customers is None:
    customers = raw

if customers is None:
    st.error(
        "No customer dataset was found. Check that the CSV files "
        "are included in the data folder."
    )
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Customer Data",
        "Customer Segments",
        "Visualizations",
        "Project Information",
    ],
)

# Try to identify the cluster column automatically.
cluster_column = next(
    (
        column
        for column in customers.columns
        if any(
            word in column.lower()
            for word in ["cluster", "segment", "group"]
        )
    ),
    None,
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.header("Project Overview")

    total_customers = len(customers)
    total_columns = len(customers.columns)

    col1, col2, col3 = st.columns(3)

    col1.metric("Customer Records", f"{total_customers:,}")
    col2.metric("Dataset Features", total_columns)

    if cluster_column:
        cluster_count = customers[cluster_column].nunique()
        col3.metric("Customer Segments", cluster_count)
    else:
        col3.metric("Customer Segments", "See results")

    st.markdown("### Dataset Preview")
    st.dataframe(customers.head(10), use_container_width=True)

    st.markdown("### Dataset Summary")
    st.write(
        f"The selected dataset contains {total_customers:,} rows "
        f"and {total_columns} columns."
    )

    with st.expander("View descriptive statistics"):
        st.dataframe(
            customers.describe(include="all").transpose(),
            use_container_width=True,
        )

# --------------------------------------------------
# Customer Data
# --------------------------------------------------
elif page == "Customer Data":
    st.header("Customer Dataset")

    dataset_choice = st.selectbox(
        "Select dataset",
        [
            "Available customer results",
            "Raw dataset",
            "Cleaned dataset",
        ],
    )

    selected_data = {
        "Available customer results": customers,
        "Raw dataset": raw,
        "Cleaned dataset": cleaned,
    }[dataset_choice]

    if selected_data is None:
        st.warning("This dataset file is not available.")
    else:
        st.write(f"Records: {len(selected_data):,}")
        st.dataframe(selected_data, use_container_width=True)

        csv_data = selected_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="customer_data.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# Customer Segments
# --------------------------------------------------
elif page == "Customer Segments":
    st.header("Customer Segmentation Results")

    if cluster_column is None:
        st.warning(
            "A cluster or segment column was not detected. "
            "Check the columns in customer_segments.csv."
        )
    else:
        st.write("Customer counts by segment")

        counts = (
            customers[cluster_column]
            .value_counts(dropna=False)
            .sort_index()
        )

        st.bar_chart(counts)

        st.dataframe(
            counts.rename("Customer Count").reset_index(),
            use_container_width=True,
        )

        st.markdown("### Segment Profiles")

        numeric_columns = customers.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            profiles = customers.groupby(
                cluster_column,
                dropna=False,
            )[numeric_columns].mean(numeric_only=True)

            st.dataframe(profiles, use_container_width=True)

        st.markdown("### Customer Records")
        chosen_cluster = st.selectbox(
            "Filter by segment",
            ["All segments"] + list(
                customers[cluster_column].dropna().unique()
            ),
        )

        if chosen_cluster == "All segments":
            filtered = customers
        else:
            filtered = customers[
                customers[cluster_column] == chosen_cluster
            ]

        st.dataframe(filtered, use_container_width=True)

# --------------------------------------------------
# Visualizations
# --------------------------------------------------
elif page == "Visualizations":
    st.header("Project Visualizations")

    if not OUTPUT_DIR.exists():
        st.warning("The outputs folder was not found.")
    else:
        image_files = sorted(
            list(OUTPUT_DIR.glob("*.png"))
            + list(OUTPUT_DIR.glob("*.jpg"))
            + list(OUTPUT_DIR.glob("*.jpeg"))
        )

        if not image_files:
            st.info("No saved chart images were found.")
        else:
            for image_path in image_files:
                with st.expander(
                    image_path.stem.replace("_", " ").title(),
                    expanded=False,
                ):
                    st.image(
                        str(image_path),
                        caption=image_path.stem.replace("_", " ").title(),
                        use_container_width=True,
                    )

# --------------------------------------------------
# Project Information
# --------------------------------------------------
elif page == "Project Information":
    st.header("About This Project")

    st.markdown(
        """
        **Objective:** Group customers into meaningful segments
        based on their characteristics and spending behavior.

        **Methodology**
        1. Load and inspect customer data.
        2. Clean and prepare relevant features.
        3. Apply K-Means clustering.
        4. Analyze cluster profiles and customer behavior.
        5. Visualize and interpret the results.

        **Technologies**
        - Python and Pandas
        - Scikit-learn
        - Matplotlib and Seaborn
        - Streamlit

        **Potential applications**
        - Customer targeting
        - Marketing campaign planning
        - Spending behavior analysis
        - Customer group comparison
        """
    )

    report_path = (
        BASE_DIR
        / "report"
        / "Customer_Segmentation_Using_Data_Science_Report.pdf"
    )

    if report_path.exists():
        with open(report_path, "rb") as report_file:
            st.download_button(
                "Download Project Report (PDF)",
                data=report_file.read(),
                file_name=report_path.name,
                mime="application/pdf",
            )

st.divider()
st.caption("Customer Segmentation Using Data Science · Internship Project")
