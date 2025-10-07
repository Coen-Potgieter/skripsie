import pandas as pd
import numpy as np
from pandas.core.api import DataFrame
from scipy.stats import gamma, norm
import os
import pandas as pd
import folium


def save_dataframe(df: pd.DataFrame, csv_path: str, debug=False):
    """
    save dataframe to a csv file, creating directories if needed.

    args:
        df: dataframe to save
        csv_path: path where csv file will be saved
        debug: whether to print debug information

    returns:
        none
    """

    # extract directory path from the full file path
    directory = os.path.dirname(csv_path)

    # create directories if they don't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        if debug:
            print(f"created directory: `{directory}`")

    # save the dataframe
    df.to_csv(csv_path, index=False)

    if debug:
        print(f"saved csv to: `{csv_path}`")


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
    df["SPI"] = spi_values

    # Clean up helper column
    df.drop(columns=["precip_roll"], inplace=True)

    # Add Month Column
    df = add_month_name(df)

    return df


def add_month_name(df: pd.DataFrame):
    # Add month abbreviation column
    df["month_abbr"] = df["date"].dt.strftime("%b").str.upper()
    return df


def plot_stations(df: pd.DataFrame, file_path: str):

    df = df[["NameUsed", "lat", "lon"]]
    print(df)

    # Take out stations with NaNs
    df = df.dropna()

    # Assuming your dataframe is named 'df'
    # First, we need to convert the lat/lon from string format with commas to floats
    df["lat"] = df["lat"].astype(str).str.replace(",", ".").astype(float)
    df["lon"] = df["lon"].astype(str).str.replace(",", ".").astype(float)

    # Create a base map centered on the mean coordinates
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    # Add markers for each station
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=row["NameUsed"],
            tooltip=row["NameUsed"],
        ).add_to(m)

    # Save the map
    m.save(file_path)
    print(f"Interactive map saved as '{file_path}'")


def messing_around(df: pd.DataFrame):

    # Assuming your DataFrame is named 'df'
    df = df[df["StationName"] == "Krantz Kloof @ Korinte-Vet Dam"]
    print(df[["StationName", "lat", "lon"]])


def get_stations_by_area(area_coords):

    # Extract Meta Data Of Stations
    meta_df = pd.read_csv("./data/WRZ2019meta.csv")

    # Convert lat column to numeric (handling comma as decimal separator)
    meta_df["lat"] = meta_df["lat"].astype(str).str.replace(",", ".").astype(float)
    meta_df["lon"] = meta_df["lon"].astype(str).str.replace(",", ".").astype(float)

    # Filter dataframe based on coordinates
    filtered_df = meta_df[
        (meta_df["lat"] >= area_coords["lat_min"])
        & (meta_df["lat"] <= area_coords["lat_max"])
        & (meta_df["lon"] >= area_coords["lon_min"])
        & (meta_df["lon"] <= area_coords["lon_max"])
        & (meta_df["Source"] == "DWS")
        & (meta_df["NameUsed"] != "VREDENDALD")
    ]

    # Some manual filtering cause fuck this data set...
    fix_dict = {
        "WELLINGTOND": "WELLINGTON",
    }
    filtered_df["NameUsed"] = filtered_df["NameUsed"].replace(fix_dict)
    return filtered_df


def avg_spi_by_station_names(station_names, k=3):
    df = pd.read_csv("./data/DWSmon.csv")

    # The most performant way is to append each small DataFrame to a list during the loop,
    #   and then convert that list into a single large DataFrame after the loop completes using pd.concat().
    list_of_dfs = []
    for name in station_names:

        station_df = df[df["station"] == name]
        station_spi = calculate_spi_station(station_df, scale=k)
        list_of_dfs.append(station_spi)

    # Convert List To Big DataFrame
    big_spi_df = pd.concat(list_of_dfs)

    # Group by year and month, calculate mean SPI, and rename to SDI
    result_df = big_spi_df.groupby(["year", "month"], as_index=False)["SPI"].mean()

    # Keep only the required columns
    result_df = result_df[["year", "month", "SPI"]]

    print(result_df.head(20))
    print(f"\nShape: {result_df.shape}")
    print(f"\nTotal rows: {len(result_df)}")

    return result_df


def validate_spi(df: pd.DataFrame) -> None:
    """
    Validate SPI dataframe with columns: year, month, SPI.
    Prints validation results to console.
    """
    print("SPI Data Validation Report")
    print("=" * 35)

    # 1. Column check
    expected_cols = {"year", "month", "SPI"}
    print(f"Has expected columns: {set(df.columns) >= expected_cols}")

    # 2. Range check (SPI typically ranges from -3 to +3, but can exceed)
    min_val = df["SPI"].min()
    max_val = df["SPI"].max()
    print(
        f"SPI values typically between -3 & 3. They are between: [{min_val:.4f}, {max_val:.4f}]"
    )
    extreme_count = ((df["SPI"] < -3) | (df["SPI"] > 3)).sum()
    print(
        f"Values outside [-3, 3] range: {extreme_count} ({extreme_count/len(df)*100:.1f}%)"
    )

    # 3. Temporal order check
    ordered = df.sort_values(["year", "month"]).equals(df)
    print(f"Data ordered by year/month: {ordered}")

    # 4. Mean sanity (SPI should have mean close to 0)
    mean_val = df["SPI"].mean()
    print(f"Mean SPI: {mean_val:.3f} (should be close to 0: {-0.5 <= mean_val <= 0.5})")

    # 5. Standard deviation check (SPI should have std close to 1)
    std_val = df["SPI"].std()
    print(f"Std Dev SPI: {std_val:.3f} (should be close to 1: {0.8 <= std_val <= 1.2})")

    # 6. Missing values
    has_missing = df.isna().any().any()
    if has_missing:
        print("Data Frame Has Missing Values...")
        nan_df = df[df.isnull().any(axis=1)]
        for index, row in nan_df.iterrows():
            print(
                f"\tNaN Record: #{index}: (year, month) = ({int(row['year'])}, {int(row['month'])})"
            )
        print("\tImputing With Mean Value...")
        df = df.fillna(df.mean())
        print(f"\tNo missing values: {not df.isna().any().any()}")
    else:
        print(f"\tNo missing values: True")

    # 7. Drought/wet condition distribution
    if not df.empty and not df["SPI"].isna().all():
        drought_severe = (df["SPI"] <= -1.5).sum()
        drought_moderate = ((df["SPI"] > -1.5) & (df["SPI"] <= -1.0)).sum()
        normal = ((df["SPI"] > -1.0) & (df["SPI"] < 1.0)).sum()
        wet_moderate = ((df["SPI"] >= 1.0) & (df["SPI"] < 1.5)).sum()
        wet_severe = (df["SPI"] >= 1.5).sum()

        total = len(df)
        print(f"Condition distribution:")
        print(
            f"\tSevere drought (SPI ≤ -1.5): {drought_severe} ({drought_severe/total*100:.1f}%)"
        )
        print(
            f"\tModerate drought (-1.5 < SPI ≤ -1.0): {drought_moderate} ({drought_moderate/total*100:.1f}%)"
        )
        print(f"\tNormal (-1.0 < SPI < 1.0): {normal} ({normal/total*100:.1f}%)")
        print(
            f"\tModerately wet (1.0 ≤ SPI < 1.5): {wet_moderate} ({wet_moderate/total*100:.1f}%)"
        )
        print(f"\tSeverely wet (SPI ≥ 1.5): {wet_severe} ({wet_severe/total*100:.1f}%)")
    else:
        print("Condition distribution skipped (no valid data)")

    print("=" * 35)


def main():
    # ==================== Pick CSV File & Import ====================
    bad2_csv = "./data/10dayrain.csv"
    bad3_csv = "./data/Mon_dwb.csv"
    station_meta_data_csv = "./data/WRZ2019meta.csv"
    dws_daily_rainfall_csv = "./data/DWS2019.csv"
    bad1_csv = "./data/mm2019.csv"
    bad4_csv = "./data/SAWS_cumulates.csv"
    dws_monthly_rainfall_csv = "./data/DWSmon.csv"

    # data = pd.read_csv(station_meta_data_csv)
    # data2 = pd.read_csv(dws_monthly_rainfall_csv)
    # print(data[data["NameUsed"] == "WELLINGTOND"])
    # print(data[data["NameUsed"] == "VREDENDALD"])
    # print(data2)
    # print(data2[data2["station"] == "VREDENDAL"])
    # return

    # ==================== Extract Stations By Study Area ====================
    study_area_coords = {
        "lat_max": -30.7,
        "lat_min": -34.83,
        "lon_min": 17.85,
        "lon_max": 21.17,
    }

    stations_in_study_area = get_stations_by_area(study_area_coords)
    # plot_stations(stations_in_study_area, "study_area.html")

    # return

    # ==================== Exrtact by Station Names ====================
    # Need a list of station names here
    station_names = stations_in_study_area["NameUsed"].to_list()
    avg_spi_df = avg_spi_by_station_names(station_names, k=3)

    validate_spi(avg_spi_df)

    save_dataframe(avg_spi_df, "./data/processed/spi.csv", debug=True)

    return

    # ==================== Playground ====================
    # messing_around(pd.read_csv(station_meta_data_csv))
    # return

    # ==================== Plot Stations ====================
    station_meta_data = pd.read_csv(station_meta_data_csv)
    plot_stations(station_meta_data, "all_stations.html")
    return

    # ==================== Filter to a single station ====================
    buffeljags_df = data[data["station"] == "BUFFELJAGS"]

    # ==================== Calculate 3-month SPI ====================
    buffeljags_spi = calculate_spi_station(buffeljags_df, scale=3)

    save_dataframe(buffeljags_spi, "./data/processed/spi.csv", debug=True)
    return


if __name__ == "__main__":
    main()
