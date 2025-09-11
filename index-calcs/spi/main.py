import pandas as pd
import numpy as np
from scipy.stats import gamma, norm

def save_dataframe(df: pd.DataFrame, csv_path: str, debug=False):
    """
    Save DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        csv_path: Path where CSV file will be saved
    
    Returns:
        None
    """

    df.to_csv(csv_path, index=False)

    if debug:
        print(f"Saved CSV to: `{csv_path}`")

def get_nan_rain_values(data: pd.DataFrame, station):
    station_data = data[data["station"] == station]
    nan_instances = []
    for i in range(station_data.shape[0]):

        if pd.isna(station_data.iloc[i, 3]):
            nan_instances.append(
                (
                    int(station_data.iloc[i, 0]),
                    int(station_data.iloc[i, 1]),
                )
            )
    return nan_instances


def get_stations(data: pd.DataFrame):
    seenStations = []
    all_stations = data.loc[:, "station"].to_list()

    for station in all_stations:
        if station in seenStations:
            continue

        seenStations.append(station)

    return seenStations


def calculate_spi_station(df, scale=3):
    """
    Calculate SPI for a given station's monthly rainfall data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['year', 'month', 'station', 'rain'].
        Data should be for ONE station only.
    scale : int
        Aggregation scale in months (e.g., 3 for 3-month SPI).

    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'SPI_<scale>' column.
    """

    # Make a datetime column
    df = df.copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values(by="date")

    # Rolling sum for aggregation
    df["precip_roll"] = df["rain"].rolling(scale, min_periods=scale).sum()

    # Prepare SPI column
    spi_values = np.full(len(df), np.nan)

    # Calculate SPI month-wise
    for month in range(1, 13):
        month_mask = df["date"].dt.month == month
        data_month = df.loc[month_mask, "precip_roll"].dropna()

        if len(data_month) < 3:
            continue

        # Probability of zero rainfall
        q = np.sum(data_month == 0) / len(data_month)

        # Fit gamma distribution (only on nonzero values)
        nonzero_data = data_month[data_month > 0]
        if len(nonzero_data) < 3:
            continue

        shape, loc, scale_param = gamma.fit(nonzero_data, floc=0)

        # Compute CDF and transform to SPI
        cdf_vals = []
        for val in data_month:
            if val > 0:
                G = gamma.cdf(val, shape, loc=0, scale=scale_param)
                H = q + (1 - q) * G
            else:
                H = q
            cdf_vals.append(H)

        # Convert to standard normal quantiles
        spi_month = norm.ppf(cdf_vals)

        # Assign values back into the SPI array
        spi_values[np.where(month_mask)[0][-len(spi_month) :]] = spi_month

    # Store SPI
    df[f"SPI_{scale}"] = spi_values

    # Clean up helper column
    df.drop(columns=["precip_roll"], inplace=True)

    # Add Month Column
    df = add_month_name(df)

    return df


def add_month_name(df: pd.DataFrame):
    # Add month abbreviation column
    df['month_abbr'] = df['date'].dt.strftime('%b').str.upper()
    return df



def main():
    # ==================== Pick CSV File ====================
    bad2 = "./data/10dayrain.csv"
    bad3 = "./data/Mon_dwb.csv"
    station_meta_data = "./data/WRZ2019meta.csv"
    dws_daily_rainfall = "./data/DWS2019.csv"
    bad1 = "./data/mm2019.csv"
    bad4 = "./data/SAWS_cumulates.csv"

    dws_monthly_rainfall = "./data/DWSmon.csv"

    # ==================== Filter to a single station ====================
    data = pd.read_csv(dws_monthly_rainfall)
    buffeljags_df = data[data["station"] == "BUFFELJAGS"]

    # ==================== Calculate 3-month SPI ====================
    buffeljags_spi = calculate_spi_station(buffeljags_df, scale=3)

    save_dataframe(buffeljags_spi, "./../combine_indices/data/buffeljags_spi.csv", debug=True)

    return

if __name__ == "__main__":
    main()
