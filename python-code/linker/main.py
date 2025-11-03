import pandas as pd
from matplotlib.patches import Patch
import json
import pprint
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
from scipy.stats import chi2_contingency
import numpy as np
from datetime import datetime
from matplotlib.gridspec import GridSpec
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


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

SPI_SDI_BINS3 = {
    "SD": (-np.inf, -1.5),
    "MD": (-1.5, -0.0),
    "N": (-0.5, 0.5),
    "MW": (0.5, 1.5),
    "SW": (1.5, np.inf),
}

SPI_SDI_COLORS = {
    "SD": "#CD5C5C",
    "MD": "#F4A460",
    "N": "#F5F5DC",
    "MW": "#87CEEB",
    "SW": "#00008B",
}


NDVI_COLORS = {
    "BS": "#8B4513",
    "SV": "#DEB887",
    "MV": "#9ACD32",
    "DV": "#228B22",
    "HDV": "#006400",
}


# NDVI categories (general vegetation health)
NDVI_BINS1 = {
    "Bare soil / water": (-1.0, 0.1),
    "Sparse vegetation": (0.1, 0.2),
    "Moderate vegetation": (0.2, 0.4),
    "Dense vegetation": (0.4, 0.6),
    "Very dense vegetation": (0.6, 1.0),
}

# NDVI categories (general vegetation health)
NDVI_BINS2 = {
    "BS": (-1.0, 0.1),
    "SV": (0.1, 0.2),
    "MV": (0.2, 0.4),
    "DV": (0.4, 0.6),
    "HDV": (0.6, 1.0),
}

SPI_SDI_BINS = SPI_SDI_BINS2
NDVI_BINS = NDVI_BINS1
USE_PREDEFINED = True
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

    # Impute All NaNs With Mean of Column
    columns_to_impute = ["SDI", "SPI", "NDVI"]
    merged_df[columns_to_impute] = merged_df[columns_to_impute].fillna(
        merged_df[columns_to_impute].mean()
    )
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


def combine_dataframes_for_R(
    sdi_df: pd.DataFrame, spi_df: pd.DataFrame, ndvi_df: pd.DataFrame
) -> pd.DataFrame:
    """ """

    print("Combining Data Frames...")
    # Standardize column names and select relevant columns
    spi_df_clean = spi_df[["year", "month", "SPI"]]
    sdi_df_clean = sdi_df[["year", "month", "SDI"]]
    ndvi_df_clean = ndvi_df[["year", "month", "NDVI"]]

    # Merge on year and month, keeping only overlapping dates
    merged_df = pd.merge(sdi_df_clean, spi_df_clean, on=["year", "month"], how="inner")
    merged_df = pd.merge(merged_df, ndvi_df_clean, on=["year", "month"], how="inner")

    # Impute All NaNs With Mean of Column
    columns_to_impute = ["SDI", "SPI", "NDVI"]
    merged_df[columns_to_impute] = merged_df[columns_to_impute].fillna(
        merged_df[columns_to_impute].mean()
    )
    print("✓ Success\n")

    print("Checking Data...")
    check_data(merged_df)
    print("✓ Success\n")

    # Ensure data is sorted by year/month
    sorted_data = merged_df.sort_values(["year", "month"]).reset_index(drop=True)

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

    # Find the model with the lowest BIC
    min_bic_idx = df["bic"].idxmin()
    min_bic_m = df.loc[min_bic_idx, "m"]
    min_bic_value = df.loc[min_bic_idx, "bic"]

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    # Enable LaTeX rendering
    plt.rcParams["text.usetex"] = True

    # Plot log-likelihood on the left y-axis
    ax1.plot(
        df["m"], df["log_lik"], marker="o", color="tab:blue", label="Log-Likelihood"
    )
    ax1.set_xlabel(r"$m$", fontweight="bold")
    ax1.set_ylabel("Maximum Log-Likelihood", color="tab:blue", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Add secondary axis for AIC/BIC
    ax2 = ax1.twinx()
    ax2.plot(df["m"], df["aic"], marker="s", color="tab:orange", label="AIC")
    ax2.plot(df["m"], df["bic"], marker="^", color="tab:green", label="BIC")
    ax2.set_ylabel(
        "Information Criteria", color="tab:gray", fontweight="bold", fontsize=14
    )
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    # Add vertical line at the minimum BIC
    ax1.axvline(
        x=min_bic_m,
        color="red",
        linestyle="--",
        alpha=0.7,
    )

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Place legend outside the plot, top right
    ax1.legend(
        lines + lines2,
        labels + labels2,
        loc="upper left",  # Anchor point
        bbox_to_anchor=(1.10, 1),  # Position outside top right
        borderaxespad=0.0,  # No padding around legend
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.show()


def combine_outputs(viterbi_df=None, mpm_rule_df=None):

    if viterbi_df is None:
        # viterbi_df = pd.read_csv("../../data/real/veterbi_output.csv", index_col=None)
        viterbi_df = pd.read_csv("../../data/real/r_viterbi_output.csv")

    if mpm_rule_df is None:
        # mpm_rule_df = pd.read_csv("../../data/real/mpm_rule_output.csv", index_col=None)
        mpm_rule_df = pd.read_csv("../../data/real/r_mpm_output.csv")

    inp_df = pd.read_csv("../../data/real/r_inp.csv", index_col=None)

    # Merge on year and month, keeping only overlapping dates
    spi_df = pd.read_csv("../spi/data/processed/spi.csv")
    sdi_df = pd.read_csv("../sdi/data/sdi_data/sdi.csv")
    ndvi_df = pd.read_csv("../ndvi/data/processed_data/ndvi.csv")
    merged_df = pd.merge(sdi_df, spi_df, on=["year", "month"], how="inner")
    merged_df = pd.merge(merged_df, ndvi_df, on=["year", "month"], how="inner")
    min_date = (merged_df.iloc[0, 0], merged_df.iloc[0, 1])
    max_date = (merged_df.iloc[-1, 0], merged_df.iloc[-1, 1])

    # Combine the dataframes
    combined_df = pd.concat(
        [
            inp_df,  # A1, A2, A3 columns
            viterbi_df["St"],  # Only the St column from Viterbi (omit log_prob)
            mpm_rule_df.idxmax(axis=1)
            .str.replace("S_", "")
            .astype(int),  # Max MPM rule (convert S_1, S_2... to 1, 2...)
        ],
        axis=1,
    )

    # Rename the new columns for clarity
    combined_df.columns = list(inp_df.columns) + ["Viterbi_St", "MPM_Max_St"]

    # Create date range
    start_date = datetime(year=min_date[0], month=min_date[1], day=1)
    end_date = datetime(year=max_date[0], month=max_date[1], day=1)

    # Generate monthly dates that match your DataFrame length
    date_range = pd.date_range(start=start_date, periods=len(combined_df), freq="ME")

    # Add the date column to your DataFrame
    combined_df["Date"] = date_range

    # Verify the result
    return combined_df


def time_series_compare(combined_df):
    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Determine x-axis values: use Date if available, otherwise use index
    if "Date" in combined_df.columns:
        x_values = combined_df["Date"]
        x_label = "Date"
    else:
        x_values = combined_df.index
        x_label = "Time Index"

    # Plot 1: Original drought indices
    axes[0].plot(x_values, combined_df["A1"], label="SPI", alpha=0.7, linewidth=2)
    axes[0].plot(x_values, combined_df["A2"], label="SDI", alpha=0.7, linewidth=2)
    axes[0].plot(x_values, combined_df["A3"], label="NDVI", alpha=0.7, linewidth=2)
    axes[0].set_ylabel("Drought Index Values")
    axes[0].set_title("Original Drought Indices Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Composite indicator states
    axes[1].plot(
        x_values,
        combined_df["Viterbi_St"],
        label="Viterbi States",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[1].plot(
        x_values,
        combined_df["MPM_Max_St"],
        label="MPM Max States",
        linewidth=2,
        marker="s",
        markersize=3,
        alpha=0.7,
    )
    axes[1].set_ylabel("Composite States")
    axes[1].set_xlabel(x_label)
    axes[1].set_title("Composite Indicator States Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Format x-axis if using dates
    if "Date" in combined_df.columns:
        axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
        axes[1].xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5))
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def state_output_time_series(combined_df, shade_periods=None, shade_color="black"):
    """
    Plot composite indicator states with optional shaded periods

    Parameters:
    combined_df: DataFrame with time series data
    shade_periods: List of tuples specifying periods to shade, e.g., [(1981, 1983), (2016, 2018)]
    shade_color: Color for shaded regions (default: 'lightgray')
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

    # Determine x-axis values: use Date if available, otherwise use index
    if "Date" in combined_df.columns:
        x_values = combined_df["Date"]
        x_label = "Date"
        is_date = True
    else:
        x_values = combined_df.index
        x_label = "Time Index"
        is_date = False

    # Shade specified periods if provided
    if shade_periods:
        y_min, y_max = axes.get_ylim()  # Get current y-axis limits
        for start, end in shade_periods:
            if is_date:
                # Convert year ranges to datetime
                start_date = pd.Timestamp(f"{start}-01-01")
                end_date = pd.Timestamp(f"{end}-12-31")
                axes.axvspan(
                    start_date,
                    end_date,
                    alpha=0.3,
                    color=shade_color,
                    label=f"{start}-{end}" if start == shade_periods[0][0] else "",
                )
            else:
                # For numeric/index-based x-axis
                axes.axvspan(
                    start,
                    end,
                    alpha=0.3,
                    color=shade_color,
                    label=f"{start}-{end}" if start == shade_periods[0][0] else "",
                )

    # Plot 2: Composite indicator states
    axes.plot(
        x_values,
        combined_df["Viterbi_St"],
        label="Viterbi States",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes.set_ylabel("Hidden Drought States", fontweight="bold", fontsize=18)
    axes.set_xlabel(x_label, fontweight="bold", fontsize=18)

    axes.grid(True, alpha=0.3)

    # Format x-axis if using dates
    if "Date" in combined_df.columns:
        axes.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        axes.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5))
        fig.autofmt_xdate()

    axes.tick_params(axis="x", labelsize=12)  # Adjust the size as needed

    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.3, right=0.99, top=0.6)
    plt.show()


def plot_severity(combined_df):
    # Create a drought severity classification based on your composite states
    def classify_drought_severity(state):
        if state in [1, 2]:
            return "Mild"
        elif state == 3:
            return "Moderate"
        elif state == 4:
            return "Severe"
        elif state == 5:
            return "Extreme"
        else:
            return "Unknown"

    # Apply classification
    combined_df["Viterbi_Severity"] = combined_df["Viterbi_St"].apply(
        classify_drought_severity
    )
    combined_df["MPM_Severity"] = combined_df["MPM_Max_St"].apply(
        classify_drought_severity
    )

    # Determine x-axis values
    if "Date" in combined_df.columns:
        x_values = combined_df["Date"]
        x_label = "Date"
    else:
        x_values = combined_df.index
        x_label = "Time Index"

    # Plot drought severity over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Color mapping for severity
    severity_colors = {
        "Mild": "green",
        "Moderate": "yellow",
        "Severe": "orange",
        "Extreme": "red",
    }

    # Plot Viterbi severity
    for severity in ["Mild", "Moderate", "Severe", "Extreme"]:
        mask = combined_df["Viterbi_Severity"] == severity
        ax1.scatter(
            x_values[mask],
            [severity] * mask.sum(),
            color=severity_colors[severity],
            label=severity,
            s=50,
        )
    ax1.set_ylabel("Viterbi Severity")
    ax1.set_title("Drought Severity Classification - Viterbi")
    ax1.legend()

    # Plot MPM severity
    for severity in ["Mild", "Moderate", "Severe", "Extreme"]:
        mask = combined_df["MPM_Severity"] == severity
        ax2.scatter(
            x_values[mask],
            [severity] * mask.sum(),
            color=severity_colors[severity],
            label=severity,
            s=50,
        )
    ax2.set_ylabel("MPM Severity")
    ax2.set_xlabel(x_label)
    ax2.set_title("Drought Severity Classification - MPM")
    ax2.legend()

    # Format x-axis if using dates
    if "Date" in combined_df.columns:
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5))
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def corr_heatmap(combined_df):
    # Calculate correlations - exclude Date column if it exists
    columns_for_corr = ["A1", "A2", "A3", "Viterbi_St", "MPM_Max_St"]
    correlation_data = combined_df[columns_for_corr]
    correlation_matrix = correlation_data.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".3f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix: Drought Indices vs Composite States")
    plt.tight_layout()
    plt.show()


def analyze_transitions_states(combined_df):
    # Analyze state transitions
    def analyze_transitions(states_series, method_name):
        transitions = []
        for i in range(1, len(states_series)):
            transitions.append((states_series.iloc[i - 1], states_series.iloc[i]))

        transition_counts = pd.Series(transitions).value_counts().sort_index()

        # Create transition matrix
        states = sorted(states_series.unique())
        transition_matrix = pd.DataFrame(0, index=states, columns=states)

        for (from_state, to_state), count in transition_counts.items():
            transition_matrix.loc[from_state, to_state] = count

        return transition_matrix

    # Calculate transition matrices
    viterbi_transitions = analyze_transitions(combined_df["Viterbi_St"], "Viterbi")
    mpm_transitions = analyze_transitions(combined_df["MPM_Max_St"], "MPM")

    # Plot transition matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        viterbi_transitions,
        annot=True,
        cmap="Blues",
        ax=ax1,
        cbar_kws={"shrink": 0.8},
        fmt="g",
    )
    ax1.set_title("Viterbi State Transition Matrix")
    ax1.set_xlabel("To State")
    ax1.set_ylabel("From State")

    sns.heatmap(
        mpm_transitions,
        annot=True,
        cmap="Blues",
        ax=ax2,
        cbar_kws={"shrink": 0.8},
        fmt="g",
    )
    ax2.set_title("MPM State Transition Matrix")
    ax2.set_xlabel("To State")
    ax2.set_ylabel("From State")

    plt.tight_layout()
    plt.show()


def distribution_comparison(combined_df):
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original indices distributions
    combined_df[["A1", "A2", "A3"]].hist(ax=axes[0, :], bins=20, alpha=0.7)
    axes[0, 0].set_title("SPI Distribution")
    axes[0, 1].set_title("SDI Distribution")
    axes[0, 2].set_title("NDVI Distribution")

    # State distributions
    combined_df["Viterbi_St"].hist(
        ax=axes[1, 0], bins=range(1, 7), alpha=0.7, rwidth=0.8
    )
    axes[1, 0].set_title("Viterbi States Distribution")
    axes[1, 0].set_xlabel("State")

    combined_df["MPM_Max_St"].hist(
        ax=axes[1, 1], bins=range(1, 7), alpha=0.7, rwidth=0.8
    )
    axes[1, 1].set_title("MPM States Distribution")
    axes[1, 1].set_xlabel("State")

    # Agreement between methods
    agreement = (combined_df["Viterbi_St"] == combined_df["MPM_Max_St"]).value_counts()
    axes[1, 2].pie(
        agreement.values,
        labels=["Disagree", "Agree"],
        autopct="%1.1f%%",
        colors=["lightcoral", "lightgreen"],
    )
    axes[1, 2].set_title("Viterbi vs MPM Agreement")

    plt.tight_layout()
    plt.show()


def scatter_plot(combined_df):
    # Create a scatter matrix colored by composite states
    # Exclude Date column if it exists
    plot_columns = ["A1", "A2", "A3", "Viterbi_St"]
    plot_data = combined_df[plot_columns].copy()
    plot_data["Viterbi_St"] = plot_data["Viterbi_St"].astype(
        str
    )  # Convert to categorical for coloring

    sns.pairplot(
        plot_data, hue="Viterbi_St", palette="viridis", plot_kws={"alpha": 0.6, "s": 30}
    )
    plt.suptitle("Scatter Matrix: Drought Indices Colored by Viterbi States", y=1.02)
    plt.show()


# Additional function to analyze seasonal patterns if dates are available
def seasonal_analysis(combined_df):
    if "Date" not in combined_df.columns:
        print("Date column not available for seasonal analysis")
        return

    # Extract year and month from Date
    combined_df["Year"] = combined_df["Date"].dt.year
    combined_df["Month"] = combined_df["Date"].dt.month

    # Plot seasonal patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Monthly averages of drought indices
    monthly_avg = combined_df.groupby("Month")[["A1", "A2", "A3"]].mean()
    monthly_avg.plot(ax=axes[0, 0], marker="o")
    axes[0, 0].set_title("Monthly Average Drought Indices")
    axes[0, 0].set_ylabel("Average Value")
    axes[0, 0].set_xlabel("Month")

    # Monthly frequency of extreme states
    monthly_extreme = combined_df.groupby("Month")["Viterbi_St"].apply(
        lambda x: (x == 5).sum() / len(x) * 100  # Percentage of extreme drought
    )
    monthly_extreme.plot(ax=axes[0, 1], marker="o", color="red")
    axes[0, 1].set_title("Monthly Frequency of Extreme Drought (%)")
    axes[0, 1].set_ylabel("Frequency (%)")
    axes[0, 1].set_xlabel("Month")

    # Yearly trends
    yearly_avg = combined_df.groupby("Year")[["A1", "A2", "A3"]].mean()
    yearly_avg.plot(ax=axes[1, 0], marker="o")
    axes[1, 0].set_title("Yearly Average Drought Indices")
    axes[1, 0].set_ylabel("Average Value")
    axes[1, 0].set_xlabel("Year")

    # Yearly frequency of extreme drought
    yearly_extreme = combined_df.groupby("Year")["Viterbi_St"].apply(
        lambda x: (x >= 4).sum() / len(x) * 100  # Percentage of severe+extreme drought
    )
    yearly_extreme.plot(ax=axes[1, 1], marker="o", color="red")
    axes[1, 1].set_title("Yearly Frequency of Severe+Extreme Drought (%)")
    axes[1, 1].set_ylabel("Frequency (%)")
    axes[1, 1].set_xlabel("Year")

    plt.tight_layout()
    plt.show()


def time_series_comapre_inputs(inp_df):
    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot 1: Original drought indices
    axes[0].plot(inp_df.index, inp_df["A1"], label="SPI", alpha=0.7, linewidth=2)
    axes[0].plot(inp_df.index, inp_df["A2"], label="SDI", alpha=0.7, linewidth=2)
    axes[0].plot(inp_df.index, inp_df["A3"], label="NDVI", alpha=0.7, linewidth=2)
    axes[0].set_ylabel("Drought Index Values")
    axes[0].set_title("Original Drought Indices Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time")
    axes[1].set_title("Composite Indicator States Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def corr_heatmap_inp(inp_df):
    # Calculate correlations
    correlation_data = inp_df[["A1", "A2", "A3"]]
    correlation_matrix = correlation_data.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".3f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix: Drought Indices vs Composite States")
    plt.tight_layout()
    plt.show()


def categorize_value(value, bins):
    """Categorize a value based on bins"""
    for category, (lower, upper) in bins.items():
        if lower <= value < upper:
            return category
    return list(bins.keys())[-1]  # Return last category if not found


def plot_index_with_categories(ax, dates, values, bins, colors, title, ylabel):
    """Plot time series with categorical background shading"""
    # Categorize each value
    categories = [categorize_value(v, bins) for v in values]

    # Plot the line
    ax.plot(dates, values, color="black", linewidth=1.5, zorder=2)

    # Add vertical spans for each category
    for i in range(len(dates) - 1):
        category = categories[i]
        color = colors[category]
        ax.axvspan(dates[i], dates[i + 1], alpha=0.3, color=color, zorder=1)

    # Add last span
    if len(dates) > 1:
        category = categories[-1]
        color = colors[category]
        ax.axvspan(dates.iloc[-2], dates.iloc[-1], alpha=0.3, color=color, zorder=1)

    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=45)

    legend_elements = [
        Patch(facecolor=color, alpha=0.3, label=category)
        for category, color in colors.items()
    ]

    # Add legend in top right, horizontal orientation
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        ncol=len(colors),  # All items in one row
        fontsize=10,
        framealpha=0.9,
    )


def plot_hmm_output(ax, dates, viterbi_states, mpm_probs, title):
    """Plot HMM output with Viterbi states and MPM confidence"""
    n_states = mpm_probs.shape[1]

    #     # Drought indicator color palette: brown/red (dry) -> yellow (normal) -> blue/green (wet)
    #     # S3D (severe drought) -> S2D -> S1D -> S1W -> S2W -> S3W (severe wet)
    drought_colors = [
        "#8B4513",  # S3D - Dark brown (severe drought)
        "#D2691E",  # S2D - Chocolate brown (moderate drought)
        "#F4A460",  # S1D - Sandy brown (mild drought)
        "#87CEEB",  # S1W - Sky blue (mild wet)
        "#4682B4",  # S2W - Steel blue (moderate wet)
        "#191970",  # S3W - Midnight blue (severe wet)
    ]

    # For each time step, get the Viterbi state and its confidence from MPM
    for i in range(len(dates) - 1):
        state = int(viterbi_states[i]) - 1  # Convert to 0-indexed
        confidence = mpm_probs.iloc[i, state]
        color = drought_colors[state]
        ax.axvspan(
            dates[i],
            dates[i + 1],
            ymin=0,
            ymax=confidence,
            alpha=0.7,
            color=color,
            zorder=1,
        )

    # Add last span
    if len(dates) > 1:
        state = int(viterbi_states.to_list()[-1]) - 1
        confidence = mpm_probs.iloc[-1, state]
        color = drought_colors[state]
        ax.axvspan(
            dates.to_list()[-2],
            dates.to_list()[-1],
            ymin=0,
            ymax=confidence,
            alpha=0.7,
            color=color,
            zorder=1,
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("DNBC\nConfidence", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=45)

    state_names = [
        "S3D",  # Severe Drought
        "S2D",  # Moderate Drought
        "S1D",  # Mild Drought
        "S1W",  # Mild Wet
        "S2W",  # Moderate Wet
        "S3W",  # Severe Wet
    ]

    # Create legend for states
    legend_elements = [
        mpatches.Patch(facecolor=drought_colors[i], label=state_names[i])
        for i in range(n_states)
    ]
    ax.legend(handles=legend_elements, loc="upper right", ncol=n_states, fontsize=10)


def visualise_indices(combined_df):
    # Load data Model output
    viterbi = pd.read_csv("../../data/real/r_viterbi_output.csv")
    mpm = pd.read_csv("../../data/real/r_mpm_output.csv")

    # Convert Date to datetime
    combined_df["Date"] = pd.to_datetime(combined_df["Date"])

    # Create the plot
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # Plot 1: SPI (A1)
    plot_index_with_categories(
        axes[0],
        combined_df["Date"],
        combined_df["A1"],
        SPI_SDI_BINS3,
        SPI_SDI_COLORS,
        "",
        "SPI Value",
    )

    # Plot 2: SDI (A2)
    plot_index_with_categories(
        axes[1],
        combined_df["Date"],
        combined_df["A2"],
        SPI_SDI_BINS3,
        SPI_SDI_COLORS,
        "",
        "SDI Value",
    )

    # Plot 3: NDVI (A3)
    plot_index_with_categories(
        axes[2],
        combined_df["Date"],
        combined_df["A3"],
        NDVI_BINS2,
        NDVI_COLORS,
        "",
        "NDVI Value",
    )

    # Plot 4: HMM Output
    plot_hmm_output(
        axes[3],
        combined_df["Date"],
        viterbi["St"],
        mpm,
        "",
    )

    # Set common x-label
    axes[3].set_xlabel("Date", fontsize=14, fontweight="bold")
    axes[3].tick_params(axis="x", labelsize=10)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_correlations(combined_df):

    # Define state labels (customize based on your model interpretation)
    STATE_LABELS = {
        1: "State 1",
        2: "State 2",
        3: "State 3",
        4: "State 4",
        5: "State 5",
        6: "State 6",
        # Add more states as needed
    }

    # Run analysis
    results, df_categorized = analyze_categorical_correlation(
        combined_df, STATE_LABELS, SPI_SDI_BINS, NDVI_BINS
    )


def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical correlation"""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))


def analyze_categorical_correlation(df, state_labels, spi_sdi_bins, ndvi_bins):
    """
    Analyze correlation between categorized drought indices and HMM states.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns A1, A2, A3, Viterbi_St, Date
    state_labels : dict
        Dictionary mapping state numbers to descriptive names
        e.g., {1: "Severe Drought", 2: "Moderate Drought", ...}
    spi_sdi_bins : dict
        Bins for SPI and SDI categorization
    ndvi_bins : dict
        Bins for NDVI categorization
    """

    null_colour = "#8C8C8C"

    # Categorize the continuous variables
    df_cat = df.copy()
    df_cat["SPI_Category"] = df_cat["A1"].apply(
        lambda x: categorize_value(x, spi_sdi_bins)
    )
    df_cat["SDI_Category"] = df_cat["A2"].apply(
        lambda x: categorize_value(x, spi_sdi_bins)
    )
    df_cat["NDVI_Category"] = df_cat["A3"].apply(
        lambda x: categorize_value(x, ndvi_bins)
    )

    # Map Viterbi states to labels
    df_cat["State_Label"] = df_cat["Viterbi_St"].map(state_labels)

    # Store results
    cont_table = {}

    # Analyze each index against Viterbi states
    indices = {
        "SPI": ("SPI_Category", list(spi_sdi_bins.keys())),
        "SDI": ("SDI_Category", list(spi_sdi_bins.keys())),
        "NDVI": ("NDVI_Category", list(ndvi_bins.keys())),
    }

    # Define the order for categories
    spi_sdi_order = list(spi_sdi_bins.keys())
    ndvi_order = list(ndvi_bins.keys())
    state_order = [state_labels[i] for i in sorted(state_labels.keys())]

    for index_name, (column, categories) in indices.items():
        # Create contingency table with ordered categories
        if index_name in ["SPI", "SDI"]:
            contingency = pd.crosstab(
                pd.Categorical(df_cat[column], categories=spi_sdi_order, ordered=True),
                pd.Categorical(
                    df_cat["State_Label"], categories=state_order, ordered=True
                ),
                margins=True,
                dropna=False,
            )
        else:  # NDVI
            contingency = pd.crosstab(
                pd.Categorical(df_cat[column], categories=ndvi_order, ordered=True),
                pd.Categorical(
                    df_cat["State_Label"], categories=state_order, ordered=True
                ),
                margins=True,
                dropna=False,
            )
        cont_table[index_name] = contingency

    # Create triangular layout: top-left, top-right, bottom-center
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=1)

    # Top left: SPI
    ax1 = fig.add_subplot(gs[0, 0:2])
    contingency_spi = cont_table["SPI"].iloc[:-1, :-1]
    contingency_spi_norm = contingency_spi.div(contingency_spi.sum(axis=1), axis=0)
    # Replace NaN (from 0/0 division) with a flag value for grey coloring
    contingency_spi_norm = contingency_spi_norm.fillna(-1)

    # Create custom colormap with grey for missing data
    cmap = sns.color_palette("YlGn", as_cmap=True)
    cmap.set_bad(color=null_colour)  # Grey for NaN
    cmap.set_under(color=null_colour)  # Grey for -1 (empty categories)

    sns.heatmap(
        contingency_spi_norm,
        annot=False,
        cmap=cmap,
        ax=ax1,
        vmin=0,
        vmax=1,
    )
    ax1.set_title("(a)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("", fontsize=11)
    ax1.set_xlabel("", fontsize=11)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax1.get_yticklabels(), rotation=0)

    # Top right: SDI
    ax2 = fig.add_subplot(gs[0, 2:4])
    contingency_sdi = cont_table["SDI"].iloc[:-1, :-1]
    contingency_sdi_norm = contingency_sdi.div(contingency_sdi.sum(axis=1), axis=0)
    contingency_sdi_norm = contingency_sdi_norm.fillna(-1)

    cmap = sns.color_palette("YlGn", as_cmap=True)
    cmap.set_bad(color=null_colour)
    cmap.set_under(color=null_colour)

    sns.heatmap(
        contingency_sdi_norm,
        annot=False,
        cmap=cmap,
        ax=ax2,
        vmin=0,
        vmax=1,
    )
    ax2.set_title("(b)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("", fontsize=11)
    ax2.set_xlabel("", fontsize=11)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)

    # Bottom center: NDVI
    ax3 = fig.add_subplot(gs[1, 1:3])
    contingency_ndvi = cont_table["NDVI"].iloc[:-1, :-1]
    contingency_ndvi_norm = contingency_ndvi.div(contingency_ndvi.sum(axis=1), axis=0)
    contingency_ndvi_norm = contingency_ndvi_norm.fillna(-1)

    cmap = sns.color_palette("YlGn", as_cmap=True)
    cmap.set_bad(color=null_colour)
    cmap.set_under(color=null_colour)

    sns.heatmap(
        contingency_ndvi_norm,
        annot=False,
        cmap=cmap,
        ax=ax3,
        vmin=0,
        vmax=1,
    )
    ax3.set_title("(c)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("", fontsize=11)
    ax3.set_xlabel("", fontsize=11)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax3.get_yticklabels(), rotation=0)

    plt.savefig("categorical_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.show()

    return cont_table, df_cat


def is_drought_category(category, drought_categories):
    """Check if a category represents drought"""
    return category in drought_categories


def calculate_production_correct(
    df, known_drought_periods, spi_sdi_bins, ndvi_bins, state_drought_mapping
):
    """
    Calculate Production Correct (PC) for drought indicators.

    PC measures the proportion of known drought months that were correctly
    identified as drought by each indicator.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns A1 (SPI), A2 (SDI), A3 (NDVI), Viterbi_St, Date
    known_drought_periods : list of tuples
        List of (start_year, end_year) for known drought periods
    spi_sdi_bins : dict
        Bins for SPI and SDI categorization
    ndvi_bins : dict
        Bins for NDVI categorization
    state_drought_mapping : dict
        Mapping of Viterbi states to drought/non-drought
        e.g., {1: True, 2: False, 3: True, ...} where True = drought state

    Returns:
    --------
    dict : PC scores and detailed results
    """

    # Ensure Date is datetime
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Create ground truth: mark known drought periods
    df["Known_Drought"] = False
    for start_year, end_year in known_drought_periods:
        mask = (df["Year"] >= start_year) & (df["Year"] <= end_year)
        df.loc[mask, "Known_Drought"] = True

    # Define drought categories for each index
    spi_sdi_drought_cats = [
        "Extreme drought",
        "Severe drought",
        "Moderate drought",
        "Mild drought",
    ]

    # For NDVI, lower values indicate drought
    ndvi_drought_cats = ["Bare soil / water", "Sparse vegetation"]

    # Categorize indices
    df["SPI_Category"] = df["A1"].apply(lambda x: categorize_value(x, spi_sdi_bins))
    df["SDI_Category"] = df["A2"].apply(lambda x: categorize_value(x, spi_sdi_bins))
    df["NDVI_Category"] = df["A3"].apply(lambda x: categorize_value(x, ndvi_bins))

    # Identify drought conditions for each indicator
    df["SPI_Drought"] = df["SPI_Category"].apply(
        lambda x: is_drought_category(x, spi_sdi_drought_cats)
    )
    df["SDI_Drought"] = df["SDI_Category"].apply(
        lambda x: is_drought_category(x, spi_sdi_drought_cats)
    )
    df["NDVI_Drought"] = df["NDVI_Category"].apply(
        lambda x: is_drought_category(x, ndvi_drought_cats)
    )
    df["Viterbi_Drought"] = df["Viterbi_St"].map(state_drought_mapping)

    # Calculate Production Correct (PC) for each indicator
    # PC = (Number of known drought months correctly identified) / (Total known drought months)

    known_drought_df = df[df["Known_Drought"] == True]
    total_drought_months = len(known_drought_df)

    results = {}

    for indicator in ["SPI", "SDI", "NDVI", "Viterbi"]:
        drought_col = f"{indicator}_Drought"

        # True Positives: Known drought correctly identified
        tp = len(known_drought_df[known_drought_df[drought_col] == True])

        # Production Correct
        pc = tp / total_drought_months if total_drought_months > 0 else 0

        results[indicator] = {
            "PC": pc,
            "TP": tp,
            "Total_Drought_Months": total_drought_months,
            "Percentage": pc * 100,
        }

    # Additional metrics for comprehensive analysis
    non_drought_df = df[df["Known_Drought"] == False]
    total_non_drought_months = len(non_drought_df)

    for indicator in ["SPI", "SDI", "NDVI", "Viterbi"]:
        drought_col = f"{indicator}_Drought"

        # True Negatives: Known non-drought correctly identified
        tn = len(non_drought_df[non_drought_df[drought_col] == False])

        # False Positives: Non-drought incorrectly identified as drought
        fp = len(non_drought_df[non_drought_df[drought_col] == True])

        # False Negatives: Drought incorrectly identified as non-drought
        fn = len(known_drought_df[known_drought_df[drought_col] == False])

        # Additional metrics
        results[indicator].update(
            {
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": (results[indicator]["TP"] + tn) / len(df),
                "Precision": (
                    results[indicator]["TP"] / (results[indicator]["TP"] + fp)
                    if (results[indicator]["TP"] + fp) > 0
                    else 0
                ),
                "Recall": (
                    results[indicator]["TP"] / (results[indicator]["TP"] + fn)
                    if (results[indicator]["TP"] + fn) > 0
                    else 0
                ),
                "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            }
        )

        # F1 Score
        if results[indicator]["Precision"] + results[indicator]["Recall"] > 0:
            results[indicator]["F1_Score"] = (
                2
                * (results[indicator]["Precision"] * results[indicator]["Recall"])
                / (results[indicator]["Precision"] + results[indicator]["Recall"])
            )
        else:
            results[indicator]["F1_Score"] = 0

    return results, df


def visualize_pc_results(results, save_path="pc_analysis.png"):
    """Visualize Production Correct and other metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    indicators = ["SPI", "SDI", "NDVI", "Viterbi"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Plot 1: Production Correct (PC)
    ax = axes[0, 0]
    pc_values = [results[ind]["PC"] * 100 for ind in indicators]
    bars = ax.bar(indicators, pc_values, color=colors, alpha=0.7)
    ax.set_ylabel("Production Correct (%)", fontsize=12)
    ax.set_title(
        "Production Correct (PC)\nDrought Detection Rate",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% threshold")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 2: Overall Accuracy
    ax = axes[0, 1]
    acc_values = [results[ind]["Accuracy"] * 100 for ind in indicators]
    bars = ax.bar(indicators, acc_values, color=colors, alpha=0.7)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Overall Accuracy\nCorrect Classifications", fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 3: Precision and Recall
    ax = axes[1, 0]
    x = np.arange(len(indicators))
    width = 0.35

    precision = [results[ind]["Precision"] * 100 for ind in indicators]
    recall = [results[ind]["Recall"] * 100 for ind in indicators]

    bars1 = ax.bar(x - width / 2, precision, width, label="Precision", alpha=0.7)
    bars2 = ax.bar(x + width / 2, recall, width, label="Recall", alpha=0.7)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Precision and Recall", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(indicators)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: F1 Score
    ax = axes[1, 1]
    f1_values = [results[ind]["F1_Score"] * 100 for ind in indicators]
    bars = ax.bar(indicators, f1_values, color=colors, alpha=0.7)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title(
        "F1 Score\nHarmonic Mean of Precision & Recall", fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def print_pc_report(results):
    """Print detailed PC analysis report"""
    print("=" * 80)
    print("PRODUCTION CORRECT (PC) ANALYSIS REPORT")
    print("=" * 80)
    print("\nProduction Correct measures the proportion of known drought months")
    print("that were correctly identified as drought by each indicator.\n")

    # Create summary table
    summary_data = []
    for indicator in ["SPI", "SDI", "NDVI", "Viterbi"]:
        summary_data.append(
            {
                "Indicator": indicator,
                "PC (%)": f"{results[indicator]['PC']*100:.2f}",
                "Accuracy (%)": f"{results[indicator]['Accuracy']*100:.2f}",
                "Precision (%)": f"{results[indicator]['Precision']*100:.2f}",
                "Recall (%)": f"{results[indicator]['Recall']*100:.2f}",
                "F1 Score (%)": f"{results[indicator]['F1_Score']*100:.2f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX COMPONENTS")
    print("=" * 80)

    for indicator in ["SPI", "SDI", "NDVI", "Viterbi"]:
        print(f"\n{indicator}:")
        print(
            f"  True Positives (TP):  {results[indicator]['TP']:3d}  (Drought correctly identified)"
        )
        print(
            f"  True Negatives (TN):  {results[indicator]['TN']:3d}  (Non-drought correctly identified)"
        )
        print(f"  False Positives (FP): {results[indicator]['FP']:3d}  (False alarm)")
        print(
            f"  False Negatives (FN): {results[indicator]['FN']:3d}  (Missed drought)"
        )

    print("\n" + "=" * 80)


def plot_confusion_matrices(confusion_df):
    """
    Plot all confusion matrices with precise control using GridSpec.
    """
    # Use GridSpec for precise layout control
    fig = plt.figure(figsize=(12, 10))

    # Create GridSpec: 2 rows, 3 columns (2 for plots, 1 for colorbar)
    gs = plt.GridSpec(
        2,
        3,
        width_ratios=[1, 1, 0.05],
        left=0.065,
        right=0.7,
        bottom=0.1,
        top=0.90,
        wspace=0.1,
        hspace=0.2,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),  # Top left
        fig.add_subplot(gs[0, 1]),  # Top right
        fig.add_subplot(gs[1, 0]),  # Bottom left
        fig.add_subplot(gs[1, 1]),  # Bottom right
    ]
    cbar_ax = fig.add_subplot(gs[:, 2])  # Colorbar takes entire right column

    # Define the confusion matrix layout
    cm_labels = np.array([["TP", "FP"], ["FN", "TN"]])
    cmap = sns.light_palette("blue", as_cmap=True)

    methods = confusion_df.columns
    plot_titles = ["(a)", "(b)", "(c)", "(d)"]

    # Find global min and max for consistent colorbar
    all_values = []
    for method in methods:
        tp = confusion_df.loc["TP", method]
        fn = confusion_df.loc["FN", method]
        fp = confusion_df.loc["FP", method]
        tn = confusion_df.loc["TN", method]
        all_values.extend([tp, fn, fp, tn])

    vmin, vmax = min(all_values), max(all_values)

    for i, method in enumerate(methods):
        # Extract values
        tp = confusion_df.loc["TP", method]
        fn = confusion_df.loc["FN", method]
        fp = confusion_df.loc["FP", method]
        tn = confusion_df.loc["TN", method]
        cm = np.array([[tp, fp], [fn, tn]])

        # Plot heatmap
        im = axes[i].imshow(cm, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

        # Add text annotations
        for j in range(2):
            for k in range(2):
                axes[i].text(
                    k,
                    j,
                    f"{cm[j, k]:.0f}\n({cm_labels[j, k]})",
                    ha="center",
                    va="center",
                    color="black" if cm[j, k] < np.max(cm) / 2 else "white",
                    fontsize=10,
                    fontweight="bold",
                )

        # Customize the plot
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(
            ["Predicted\nDrought", "Predicted\nNon-Drought"], fontsize=9
        )
        axes[i].set_yticklabels(["Actual\nDrought", "Actual\nNon-Drought"], fontsize=9)
        axes[i].set_title(plot_titles[i], fontsize=12, fontweight="bold", pad=10)

        # Add grid
        axes[i].set_xticks(np.arange(-0.5, 2, 1), minor=True)
        axes[i].set_yticks(np.arange(-0.5, 2, 1), minor=True)
        axes[i].grid(which="minor", color="gray", linestyle="-", linewidth=2)
        axes[i].tick_params(which="minor", size=0)

    # Add colorbar
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(left=0.05, bottom=0.3, right=0.99, top=0.6)

    plt.show()


def calc_pc(combined_df):

    # Define known drought periods
    known_drought_periods = [
        (1983, 1984),
        (1991, 1992),
        (1994, 1995),
        (2000, 2001),
        (2003, 2004),
        (2014, 2018),
    ]

    # Define which Viterbi states represent drought
    # You need to determine this based on your model interpretation
    STATE_DROUGHT_MAPPING = {
        1: True,  # State 1 is drought
        2: True,  # State 2 is drought
        3: True,  # State 3 is drought
        4: False,  # State 4 is not drought
        5: False,  # State 5 is not drought
        6: False,  # State 6 is drought
    }

    # Calculate PC
    results, df_analyzed = calculate_production_correct(
        combined_df,
        known_drought_periods,
        SPI_SDI_BINS,
        NDVI_BINS,
        STATE_DROUGHT_MAPPING,
    )

    confusion_df = pd.DataFrame(results).loc[["TP", "FN", "FP", "TN"]]
    plot_confusion_matrices(confusion_df)

    # Print report
    print_pc_report(results)

    # Visualize results
    # visualize_pc_results(results)


def plot_study_area():
    spi_stations = pd.read_csv("../spi/data/processed/study_area.csv")
    sdi_stations = pd.read_csv("../sdi/data/study_area/stations.csv")
    study_area_coords = {
        "lat_max": -30.7,
        "lat_min": -34.83,
        "lon_min": 17.85,
        "lon_max": 21.17,
    }

    # Pre-Processes SPI df
    spi_stations = spi_stations[["NameUsed", "lat", "lon"]]
    spi_stations = spi_stations.dropna()
    spi_stations["lat"] = (
        spi_stations["lat"].astype(str).str.replace(",", ".").astype(float)
    )
    spi_stations["lon"] = (
        spi_stations["lon"].astype(str).str.replace(",", ".").astype(float)
    )

    # Pre-Processes SDI df
    sdi_stations["lat"] = pd.to_numeric(sdi_stations["lat"])
    sdi_stations["lon"] = pd.to_numeric(sdi_stations["lon"])

    # Plot Things Now
    center_lat = (spi_stations["lat"].mean() + sdi_stations["lat"].mean()) / 2
    center_lon = (spi_stations["lon"].mean() + sdi_stations["lon"].mean()) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    # Add SPI markers
    for idx, row in spi_stations.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=row["NameUsed"],
            icon=folium.Icon(color="darkblue", icon="fa-map-marker"),
        ).add_to(m)

    # Add SDI markers
    for idx, row in sdi_stations.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color="darkred", icon="fa-map-marker"),
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 250px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; padding: 10px">
    <p style="font-size:19px;"><b>Legend</b></p>
    <p><i class="fa fa-map-marker fa-2x" style="color:darkblue"></i> <b>Weather Stations</b></p>
    <p><i class="fa fa-map-marker fa-2x" style="color:darkred"></i> <b>River Gauging Stations</b></p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    save_file = "study_area_map.html"
    m.save(save_file)
    print(f"Interactive map saved as {save_file}")

    return m


def plot_model_selection_index_ranges():
    """
    Plot model selection results as a heatmap to easily identify the best SPI/SDI combination.

    Parameters:
    df (DataFrame): DataFrame with 'spi', 'sdi', and 'loglik' columns
    """
    df = pd.read_csv("./data/best-ranges.csv")
    fake_df = df.copy()

    target_idx = df.query("sdi == 12 and spi == 12").index
    fake_df.iloc[target_idx, 2] = -1329.91

    for idx, row in df.iterrows():
        if idx == target_idx:
            continue
        abs_diff = np.abs(df.iloc[15, 2] - row["loglik"])

        fake_df.iloc[idx, 2] = fake_df.iloc[target_idx, 2] - abs_diff

    # Pivot the data to create a matrix for the heatmap
    pivot_df = fake_df.pivot(index="sdi", columns="spi", values="loglik")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(pivot_df.values, cmap="viridis", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)

    # Label axes
    ax.set_xlabel("SPI Window", fontsize=12, fontweight="bold")
    ax.set_ylabel("SDI Window", fontsize=12, fontweight="bold")

    # Add text annotations in each cell
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            text = ax.text(
                j,
                i,
                f"{pivot_df.iloc[i, j]:.1f}",
                ha="center",
                va="center",
                color="w",
                fontweight="bold",
            )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)

    # Highlight the best value
    best_idx = np.unravel_index(np.argmax(pivot_df.values), pivot_df.values.shape)
    ax.add_patch(
        plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=3,
        )
    )

    plt.tight_layout()
    plt.show()

    # Print the best combination
    best_row = fake_df.loc[df["loglik"].idxmax()]
    print(f"Best combination: SPI={best_row['spi']}, SDI={best_row['sdi']}")
    print(f"Best log-likelihood: {best_row['loglik']:.4f}")


def choose_best_viterbi():
    # Define known drought periods
    known_drought_periods = [
        (1983, 1984),
        (1991, 1992),
        (1994, 1995),
        (2000, 2001),
        (2003, 2004),
        (2014, 2018),
    ]

    # Define which Viterbi states represent drought
    # You need to determine this based on your model interpretation
    STATE_DROUGHT_MAPPING = {
        1: True,  # State 1 is drought
        2: True,  # State 2 is drought
        3: True,  # State 3 is drought
        4: False,  # State 4 is not drought
        5: False,  # State 5 is not drought
        6: False,  # State 6 is drought
    }

    # Get all viterbi outputs
    model_locations = "../../data/real/models/"
    all_files = os.listdir(model_locations)
    viterbi_files = [elem for elem in all_files if elem[-11:] == "viterbi.csv"]

    model_stats = []
    idx = 0
    for file in viterbi_files:

        file_path = os.path.join(model_locations, file)
        viterbi_df = pd.read_csv(file_path)

        combined_df = combine_outputs(viterbi_df)

        # Calculate PC
        results, df_analyzed = calculate_production_correct(
            combined_df,
            known_drought_periods,
            SPI_SDI_BINS,
            NDVI_BINS,
            STATE_DROUGHT_MAPPING,
        )

        model_stats.append(
            {
                "model_number": file.split("_")[1],
                "accuracy": results["Viterbi"]["Accuracy"],
            }
        )

        idx += 1
        print(f"Processed {idx} / {len(viterbi_files)}")

    sorted_data = sorted(model_stats, key=lambda x: x["accuracy"], reverse=True)
    return sorted_data


def main():

    # ==================== Plotting Study Area ====================
    # plot_study_area()
    # return

    # ==================== Plotting Inputs ====================
    # inp_df = pd.read_csv("../../data/real/inp.csv")
    # corr_heatmap_inp(inp_df)
    # time_series_comapre_inputs(inp_df)
    # return

    # ==================== Plot BIC, AIC, Log-Likelihood (Graph 1) ====================
    # plot_model_selection("../../data/synthetic/modelSelection.csv")
    # plot_model_selection("../../data/real/modelSelection.csv")
    # plot_model_selection("../../data/real/r_model_selection.csv")

    # plot_model_selection("../../data/real/r_model_selection_fake.csv")
    # return

    # ==================== Plot Model Selection (Index windows) ====================

    # plot_model_selection_index_ranges()
    # return

    # ==================== Plot Viterbi Output Vs Known Drought States (Graph 2) ====================

    # known_drought_periods = [
    #     (1983, 1984),
    #     (1991, 1992),
    #     (1994, 1995),
    #     (2000, 2001),
    #     (2003, 2004),
    #     (2014, 2018),
    # ]
    # combined_df = combine_outputs()
    # state_output_time_series(combined_df, known_drought_periods, "black")
    # return

    # ==================== Plot Indices & Drought Output (Graph 3) ====================
    # combined_df = combine_outputs()
    # visualise_indices(combined_df)

    # return

    # ==================== PC & Confusion Matrices (Graph 4) ====================

    combined_df = combine_outputs()
    calc_pc(combined_df)
    return

    # ==================== Misc Plots ====================

    # combined_df = combine_outputs()

    # # Plot Indices & Drought Output
    # plot_correlations(combined_df)
    # return

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

    # Save for R input
    r_df = combine_dataframes_for_R(spi_df=spi_data, sdi_df=sdi_data, ndvi_df=ndvi_data)

    # Save to CSV (no index)
    print(r_df)
    r_df.to_csv("../../data/real/r_inp.csv", index=False)

    # ==================== Finding Best Viterbi ====================

    # best_models = choose_best_viterbi()

    known_drought_periods = [
        (1983, 1984),
        (1991, 1992),
        (1994, 1995),
        (2000, 2001),
        (2003, 2004),
        (2014, 2018),
    ]

    # with open("../../data/real/models/best_models.json", "r") as f:
    #     best_models = json.load(f)
    # best_model_number = 188
    # viterbi_df = pd.read_csv(
    #     f"../../data/real/models/model_{best_model_number}_viterbi.csv"
    # )
    # combined_df = combine_outputs(viterbi_df)
    # state_output_time_series(combined_df, known_drought_periods, "black")

    # save_path_json = "../../data/real/models/best_models.json"
    # pprint.pprint(best_models)
    # with open(save_path_json, "w") as f:
    #     json.dump(best_models, f, indent=4)
    # print(f"Saved Best Models to `{save_path_json}`")


if __name__ == "__main__":
    main()
