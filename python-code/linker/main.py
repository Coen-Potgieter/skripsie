import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# SPI/SDI categories (standardised)
SPI_SDI_BINS1 = {
    "Extreme drought": (-np.inf, -2.0),
    "Severe drought": (-2.0, -1.5),
    "Moderate drought": (-1.5, -1.0),
    "Mild drought": (-1.0, -0.5),
    "Near normal": (-0.5, 0.5),
    "Mild wet": (0.5, 1.0),
    "Moderate wet": (1.0, 1.5),
    "Severe wet": (1.5, 2.0),
    "Extreme wet": (2.0, np.inf),
}
SPI_SDI_BINS2 = {
    "Severe drought": (-np.inf, -1.5),
    "Moderate drought": (-1.5, -0.0),
    "Normal": (-0.5, 0.5),
    "Moderate wet": (0.5, 1.5),
    "Extreme wet": (1.5, np.inf),
}


# NDVI categories (general vegetation health)
NDVI_BINS1 = {
    "Bare soil / water": (-1.0, 0.1),
    "Sparse vegetation": (0.1, 0.2),
    "Moderate vegetation": (0.2, 0.4),
    "Dense vegetation": (0.4, 0.6),
    "Very dense vegetation": (0.6, 1.0),
}

SPI_SDI_BINS = SPI_SDI_BINS2
NDVI_BINS = NDVI_BINS1
USE_PREDEFINED = False
NUM_BINS = 5
METHOD = "quantile"  # Equal-frequency bins - each bin has roughly the same number of observations. Good for balanced datasets.
# METHOD = "equal" # Equal-width bins based on min-max range. Simple and interpretable.
# METHOD = "std" # Standard deviation-based bins - specifically designed for standardized indices like SPI/SDI where values center around 0 with std ≈ 1.


def save_csv(df: pd.DataFrame, file_path):

    # Extract directory path from the full file path
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)

    df.to_csv(file_path, index=False)
    print(f"Data Frame Saved To:`{file_path}`...")


def bucketize_value(val, bins: dict) -> int:
    """Return integer code for a value given bins dict {label: (low, high)}"""
    for i, (label, (low, high)) in enumerate(bins.items()):
        if low <= val < high:
            return i + 1
    return np.nan  # fallback


def discretise_drought_indices(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert continuous SPI, SDI, NDVI into discretised integer categories.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: year, month, SPI, SDI, NDVI

    Returns:
    --------
    pd.DataFrame with discretised values
    """
    result_df = df[["year", "month"]].copy()

    if USE_PREDEFINED:
        # Original functionality with predefined bins
        if "SPI" in df.columns and SPI_SDI_BINS is not None:
            result_df["SPI"] = df["SPI"].apply(
                lambda x: bucketize_value(x, SPI_SDI_BINS)
            )

        if "SDI" in df.columns and SPI_SDI_BINS is not None:
            result_df["SDI"] = df["SDI"].apply(
                lambda x: bucketize_value(x, SPI_SDI_BINS)
            )

        if "NDVI" in df.columns and NDVI_BINS is not None:
            result_df["NDVI"] = df["NDVI"].apply(
                lambda x: bucketize_value(x, NDVI_BINS)
            )

    else:
        # Even binning based on data range
        if "SPI" in df.columns:
            if METHOD == "quantile":
                result_df["SPI"] = (
                    pd.qcut(df["SPI"], q=NUM_BINS, labels=False, duplicates="drop") + 1
                )
            elif METHOD == "equal":
                result_df["SPI"] = pd.cut(df["SPI"], bins=NUM_BINS, labels=False) + 1
            elif METHOD == "std":
                result_df["SPI"] = bin_by_std(df["SPI"], NUM_BINS) + 1

        if "SDI" in df.columns:
            if METHOD == "quantile":
                result_df["SDI"] = (
                    pd.qcut(df["SDI"], q=NUM_BINS, labels=False, duplicates="drop") + 1
                )
            elif METHOD == "equal":
                result_df["SDI"] = pd.cut(df["SDI"], bins=NUM_BINS, labels=False) + 1
            elif METHOD == "std":
                result_df["SDI"] = bin_by_std(df["SDI"], NUM_BINS) + 1

        if "NDVI" in df.columns:
            if METHOD == "quantile":
                result_df["NDVI"] = (
                    pd.qcut(df["NDVI"], q=NUM_BINS, labels=False, duplicates="drop") + 1
                )
            elif METHOD == "equal":
                result_df["NDVI"] = pd.cut(df["NDVI"], bins=NUM_BINS, labels=False) + 1
            else:
                # For NDVI with std METHOD, use equal-width as fallback
                result_df["NDVI"] = pd.cut(df["NDVI"], bins=NUM_BINS, labels=False) + 1

    # Return only columns that exist in result_df
    cols_to_return = ["year", "month"]
    for col in ["SPI", "SDI", "NDVI"]:
        if col in result_df.columns:
            cols_to_return.append(col)

    return result_df[cols_to_return]


def combine_dataframes(
    sdi_df: pd.DataFrame, spi_df: pd.DataFrame, ndvi_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine three DataFrames on year and month, keeping only SDI, SPI and NDVI columns.
    Removes null values from the beginning and end of both SDI, SPI and NDVI columns.

    Args:
        sdi_df: First DataFrame with year, month, and SDI columns
        spi_df: Second DataFrame with year, month, and SPI columns
        ndvi_df: Third DataFrame with year, month, and NDVI columns

    Returns:
        Merged DataFrame with year, month, SDI, SPI & NDVI columns
    """

    print("Combining Data Frames...")
    # Standardize column names and select relevant columns
    spi_df_clean = spi_df[["year", "month", "SPI"]]
    sdi_df_clean = sdi_df[["year", "month", "SDI"]]
    ndvi_df_clean = ndvi_df[["year", "month", "NDVI"]]

    # Merge on year and month, keeping only overlapping dates
    merged_df = pd.merge(sdi_df_clean, spi_df_clean, on=["year", "month"], how="inner")
    merged_df = pd.merge(merged_df, ndvi_df_clean, on=["year", "month"], how="inner")

    # Remove All NaNs
    merged_df = merged_df.dropna(subset=["SDI", "SPI", "NDVI"])
    print("✓ Success\n")

    print("Checking Data...")
    check_data(merged_df)
    print("✓ Success\n")

    print("Preparing Data...")
    # Discretise first
    method = METHOD
    discretised_data = discretise_drought_indices(merged_df)

    # Ensure data is sorted by year/month
    sorted_data = discretised_data.sort_values(["year", "month"]).reset_index(drop=True)

    # Select final Attributes
    final_df = sorted_data[["SPI", "SDI", "NDVI"]].rename(
        columns={"SPI": "A1", "SDI": "A2", "NDVI": "A3"}
    )
    print("✓ Success")

    return final_df


def find_missing_months(df: pd.DataFrame) -> list:
    """
    Checks a DataFrame for missing months between the earliest and latest dates.

    The DataFrame must have 'year' and 'month' columns.

    Args:
        df: The pandas DataFrame to check.

    Returns:
        A list of tuples, where each tuple contains the (year, month)
        of a missing period. Returns an empty list if there are no gaps.
    """
    # Create a 'date' column by combining 'year' and 'month'
    # The day is set to 1 as a placeholder, it doesn't affect the logic.
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )

    # Create a complete date range from the min to the max date with monthly frequency
    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS",  # 'MS' stands for Month Start frequency
    )

    # Identify the dates that are in our expected range but not in the DataFrame
    # We use the .difference() method which is efficient for this comparison.
    missing_dates = expected_dates.difference(df["date"])

    # Format the missing dates into a list of (year, month) tuples for readability
    missing_year_month = sorted([(date.year, date.month) for date in missing_dates])

    return missing_year_month


def apply_filters(df: pd.DataFrame, filters: list):
    """
    Apply multiple boolean conditions to filter a DataFrame.

    Args:
        df: Input DataFrame to filter
        filters: List of boolean conditions (e.g., [df['col'] > 0, df['col2'] == 'value'])

    Returns:
        Filtered DataFrame with rows satisfying all conditions
    """

    outp_df = df

    for filter in filters:
        outp_df = outp_df[filter]

    return outp_df


def check_data(df: pd.DataFrame):
    # Check Nans:
    if df.isna().any().any():
        print("THIS DATAFRAME HAS NaNs...")
    else:
        print("✓ No NaNs")

    # Check Continuous Data
    # Check if the DataFrame has continuous months without gaps
    missing_months = find_missing_months(df)

    if len(missing_months) != 0:
        print("✗ Missing Months:")

        for missing_date in missing_months:
            print("\t", missing_date)
    else:
        print("✓ No Missing Dates")

    # Print Range of specified columns
    target_columns = ["SPI", "SDI", "NDVI"]

    for col_name in target_columns:
        vals = df.loc[:, col_name]
        print(
            f"{col_name} Range: [{np.round(vals.min(), 2)}, {np.round(vals.max(), 2)}]"
        )


def generate_data(T: int, attribute_types: dict):
    """
    Generates synthetic data for DNBC input.

    Args:
        T (int): Number of time steps
        attribute_types (dict): Dictionary specifying attribute info.
            Example:
            {
                "A1": {"type": "discrete", "cardinality": 3},
                "A2": {"type": "continuous", "distribution": "normal", "mean": 0, "cov": 1},
                "A3": {"type": "continuous", "distribution": "normal", "mean": [0,0], "cov": [[1,0],[0,1]]}
            }

    Returns:
        pd.DataFrame: Columns -> [A1, A2, ... AN]
    """

    data = {}

    for _, (attr, spec) in enumerate(attribute_types.items(), start=1):
        if spec["type"] == "discrete":
            # Uniform categorical distribution over cardinality
            data[attr] = np.random.randint(1, spec["cardinality"] + 1, size=T)

        elif spec["type"] == "continuous":
            if spec["distribution"] == "normal":
                mean = np.array(spec["mean"])
                cov = np.array(spec["cov"])
                # If mean is scalar, treat as 1D normal
                if mean.size == 1 and cov.size == 1:
                    data[attr] = np.random.normal(loc=mean, scale=np.sqrt(cov), size=T)
                else:
                    raise ValueError(
                        "Stop trying to be clever... We not playing with multivariate normals. Remove it."
                    )
            else:
                raise ValueError(
                    f"Unsupported continuous distribution: {spec['distribution']}"
                )
        else:
            raise ValueError(f"Unsupported attribute type: {spec['type']}")

    df = pd.DataFrame(data)
    return df


def extract_output():
    data = pd.read_csv("../../data/synthetic/output.csv")
    print(data)


def plot_model_selection(csv_file: str):
    """
    Reads a CSV with columns: m, log_lik, aic, bic
    and plots log-likelihood, AIC, and BIC vs number of states (m).
    """
    # Load the data
    df = pd.read_csv(csv_file)

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot log-likelihood on the left y-axis
    ax1.plot(
        df["m"], df["log_lik"], marker="o", color="tab:blue", label="Log-Likelihood"
    )
    ax1.set_xlabel("Number of Hidden States (m)")
    ax1.set_ylabel("Log-Likelihood", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Add secondary axis for AIC/BIC
    ax2 = ax1.twinx()
    ax2.plot(df["m"], df["aic"], marker="s", color="tab:orange", label="AIC")
    ax2.plot(df["m"], df["bic"], marker="^", color="tab:green", label="BIC")
    ax2.set_ylabel("AIC / BIC", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title("Model Selection Diagnostics (Log-Lik, AIC, BIC)")
    plt.tight_layout()
    plt.show()


def playing():
    data = pd.read_csv("../../data/real/veterbi_output.csv", index_col=None)
    print(data)


def main():

    # playing()
    # return

    # ==================== Plot BIC, AIC, Log-Likelihood ====================
    # plot_model_selection("../../data/synthetic/modelSelection.csv")
    plot_model_selection("../../data/real/modelSelection.csv")
    return

    # ==================== Extract Output ====================
    # extract_output()
    # return

    # ==================== Generate Synthetic Data ====================
    # time_steps = 500
    # attrs = {
    #     "A1": {"type": "discrete", "cardinality": 8},
    #     "A2": {"type": "discrete", "cardinality": 7},
    #     "A3": {"type": "discrete", "cardinality": 3},
    # }

    # synthetic_data = generate_data(time_steps, attribute_types=attrs)
    # print(synthetic_data)
    # save_csv(synthetic_data, file_path="../../data/synthetic/test.csv")
    # return

    # ==================== Combine Data ====================

    spi_path = "../spi/data/processed/spi.csv"
    sdi_path = "../sdi/data/sdi_data/sdi.csv"
    ndvi_path = "../ndvi/data/processed_data/ndvi.csv"

    sdi_data = pd.read_csv(sdi_path)
    spi_data = pd.read_csv(spi_path)
    ndvi_data = pd.read_csv(ndvi_path)

    final_df = combine_dataframes(spi_df=spi_data, sdi_df=sdi_data, ndvi_df=ndvi_data)

    # Save to CSV (no index)
    final_df.to_csv("../../data/real/inp.csv", index=False)
    print(final_df.min())


if __name__ == "__main__":
    main()
