import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def save_csv(df: pd.DataFrame, file_path):
    df.to_csv(file_path, index=False)
    print(f"Data Frame Saved To:`{file_path}`...")


def combine_dataframes(sdi_df: pd.DataFrame, spi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two DataFrames on year and month, keeping only SDI_3 and SPI_3 columns.
    Removes null values from the beginning and end of both SDI_3 and SPI_3 columns.

    Args:
        sdi_df: First DataFrame with YEAR, MONTH, and SDI_3 columns
        spi_df: Second DataFrame with year, month, and SPI_3 columns

    Returns:
        Merged DataFrame with year, month, SDI_3, and SPI_3 columns
    """
    # Standardize column names and select relevant columns
    sdi_df_clean = sdi_df[["YEAR", "MONTH", "SDI_3"]].rename(
        columns={"YEAR": "year", "MONTH": "month"}
    )
    spi_df_clean = spi_df[["year", "month", "SPI_3"]]

    # Merge on year and month, keeping only overlapping dates
    merged_df = pd.merge(sdi_df_clean, spi_df_clean, on=["year", "month"], how="inner")

    # Remove All NaNs
    merged_df = merged_df.dropna(subset=["SDI_3", "SPI_3"])

    return merged_df


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
    target_columns = ["SPI_3", "SDI_3"]

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


def main():
    plot_model_selection("../../data/synthetic/modelSelection.csv")
    return
    # extract_output()
    # return
    attrs = {
        "A1": {"type": "discrete", "cardinality": 8},
        "A2": {"type": "discrete", "cardinality": 7},
        # Example of cts Attribute "A2": {"type": "continuous", "distribution": "normal", "mean": 0, "cov": 1},
    }

    synthetic_data = generate_data(T=10, attribute_types=attrs)
    print(synthetic_data)
    save_csv(synthetic_data, file_path="../../data/synthetic/test.csv")
    return

    spi_path = "./data/buffeljags_spi.csv"
    sdi_path = "./data/sdi.csv"

    sdi_data = pd.read_csv(sdi_path)
    spi_data = pd.read_csv(spi_path)

    combined_df = combine_dataframes(spi_df=spi_data, sdi_df=sdi_data)

    check_data(combined_df)
    # print(combined_df)


if __name__ == "__main__":
    main()
