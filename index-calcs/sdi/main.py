import warnings

warnings.filterwarnings("ignore", module="urllib3")  # Ignores all warnings from urllib3

import numpy as np
import requests as re
from bs4 import BeautifulSoup
import json
import os
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd


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

def split_date_range(start_date: datetime, end_date: datetime, max_years=20):
    start = start_date.date()
    end = end_date.date()

    pairs = []
    current_start = start

    while current_start < end:
        # Calculate next date (current_start + 20 years)
        next_date = current_start + relativedelta(years=max_years)

        # Ensure we don't exceed the end date
        if next_date > end:
            next_date = end

        pairs.append((current_start.isoformat(), next_date.isoformat()))

        # Move to the next day for the new range
        current_start = next_date + timedelta(days=1)

    return pairs


def check_valid_response(r: re.Response):
    if r.status_code != 200:
        return False
    return True


def check_valid_region(region_code: str):
    if len(region_code.strip()) != 1:
        raise ValueError(f"Given region Code: {region_code} must be a single character")

    region_code = region_code.capitalize()

    valid_region_codes = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "G",
        "J",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
    ]

    if region_code not in valid_region_codes:
        raise ValueError(
            f"Given region Code: {region_code} Is not Valid\nValid region codes are: {valid_region_codes}"
        )

    return region_code


def scrape_river_stations(region_code: str):
    region_code = check_valid_region(region_code)

    print(f"Region Code is Valid, Getting river stations in area {region_code}...")
    target_link = f"https://www.dws.gov.za/hydrology/Verified/HyStations.aspx?Region={region_code}&StationType=rbRiver"

    r = re.get(target_link)

    check_valid_response(r)

    html_content = r.content

    soup = BeautifulSoup(html_content, "html.parser")

    table = soup.find(id="tableStations")

    if table is None:
        raise ValueError(f"No table found at link: {target_link}")

    table_rows = table.find_all("tr")

    if len(table_rows) == 0:
        raise ValueError(
            f"Table found but table is empty?\nThis shouldnt be happening, so come check whats happening"
        )

    # Remove First two rows (uninformative content, see raw html if you dont believe me bro...)
    table_rows = table_rows[2:]

    print(f"Found {len(table_rows)} Stations in Area {region_code}, Now parsing...")

    stations = []
    for row in table_rows:

        cells = row.find_all("td")

        station_info = {
            "code": cells[0].find("a").text,
            "name": cells[1].text,
            "catchment_area": cells[2].text,
            "lat": cells[3].text,
            "long": cells[4].text,
            "data_avail": cells[5].text,
        }

        stations.append(station_info)

    save_dir = f"./data/station_data/river/daily/region_{region_code}.json"
    print(f"Scraped station data for region {region_code}, now saving to {save_dir}")

    # Create parent directories
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # Save the JSON file
    with open(save_dir, "w") as f:
        json.dump(stations, f, indent=4)

    print(f"Saved\n")


def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def scrape_all_station_metadata():
    valid_region_codes = [
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "G",
        "J",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
    ]

    for code in valid_region_codes:
        scrape_river_stations(code)



def text_to_dataframe(raw_text):
    """Convert raw text to DataFrame with proper error handling"""
    lines = raw_text.split("\n")
    data_lines = [line for line in lines if line[:8].isdigit()]  # Filter data rows

    processed_data = []
    for line in data_lines:
        try:
            record = {
                "DATE": pd.to_datetime(line[0:8]),
                "FLOW_RATE": (
                    float(line[10:18].strip()) if line[10:18].strip() else None
                ),
                "QUALITY": int(line[19:24].strip()),
            }
            processed_data.append(record)
        except ValueError as e:
            print(f"Warning: Skipping malformed line - {line.strip()} | Error: {e}")
            continue

    return pd.DataFrame(processed_data)


def scrape_stream_flow(station_code: str, start_date=None, end_date=None):
    # Ensure given station_code is in a region that is valid
    station_code = check_valid_region(station_code[0]) + station_code[1:]

    # Check/Find station_code in particular region
    file_path = f"./data/station_data/river/region_{station_code[0]}.json"
    with open(file_path, "r") as tf:
        all_station_data = json.load(tf)

    # Find Station in File
    station_meta_data = None
    for elem in all_station_data:
        if elem["code"] == station_code:
            station_meta_data = elem

    if station_meta_data is None:
        raise ValueError(f"Given Station: {station_code}, was not found.")

    print(f"Station {station_code} Metadata Found")

    # Get available data dates
    available_data_start = datetime.strptime(
        station_meta_data["data_avail"].split(" ")[0], "%Y-%m-%d"
    )
    available_data_end = datetime.strptime(
        station_meta_data["data_avail"].split(" ")[2], "%Y-%m-%d"
    )

    # Check if given dates are a valid format and within range of station available data (given they are not none)
    if start_date is not None:
        if not validate_date(start_date):
            raise ValueError(
                f"Invalid start_date: {start_date}. Expected format: YYYY-MM-DD"
            )

        # Since its valid, this wont fail...
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

        if available_data_start > start_date:
            print(
                f"Available Data begins @ {available_data_start.strftime('%Y-%m-%d')}, which is after given date: {start_date.strftime('%Y-%m-%d')}. Therefore getting data from {available_data_start.strftime('%Y-%m-%d')} onwards"
            )
            start_date = available_data_start
    else:
        print(
            f"No start date given, therefore scraping from the very beginning: {available_data_start.strftime('%Y-%m-%d')}"
        )
        start_date = available_data_start
    # Check if given dates are a valid format and within range of station available data (given they are not none)
    if end_date is not None:
        if not validate_date(end_date):
            raise ValueError(
                f"Invalid end_date: {end_date}. Expected format: YYYY-MM-DD"
            )

        # Since its valid, this wont fail...
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if end_date > available_data_end:
            print(
                f"Available Data ends @ {available_data_end.strftime('%Y-%m-%d')}, which is before given date: {end_date.strftime('%Y-%m-%d')}. Therefore getting data up until {available_data_end.strftime('%Y-%m-%d')}"
            )
            end_date = available_data_end
    else:
        print(
            f"No end date given, therefore scraping up until the end: {available_data_end.strftime('%Y-%m-%d')}"
        )
        end_date = available_data_end

    # Now gotta make sure we in 20 year increments
    split_dates = split_date_range(start_date, end_date, max_years=20)

    print(
        f"Scraping Data For {station_code} Between Years [{start_date.strftime('%Y-%m-%d')}, {end_date.strftime('%Y-%m-%d')}]..."
    )

    all_data = pd.DataFrame()

    for period_start, period_end in split_dates:

        print(f"Processing period: {period_start} to {period_end}")
        data_link = f"https://www.dws.gov.za/hydrology/Verified/HyData.aspx?Station={station_code}100.00&DataType=Daily&StartDT={period_start}&EndDT={period_end}&SiteType=RIV"
        r = re.get(data_link)

        if not check_valid_response(r):
            raise ValueError(f"ERROR: Got status code: {r.status_code}...")

        html_content = r.content
        soup = BeautifulSoup(html_content, "html.parser")
        raw_text = soup.find("pre").text

        # Process current batch
        current_df = text_to_dataframe(raw_text)
        # Append to combined DataFrame
        all_data = pd.concat([all_data, current_df], ignore_index=True)

    # Create file to save csv, create parent directories, and save the thing
    save_dir = f"./data/stream_flow_data/daily/{station_code}.csv"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    all_data.to_csv(save_dir, index=False)
    print(
        f"Successfully saved stream flow data for station {station_code} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} @ {save_dir}"
    )

    pass


def assess_data(df: pd.DataFrame):
    # Count NaN values in FLOW_RATE
    print("\nAssessing Nan Flow Rates...")
    nan_count = df.loc[:, "FLOW_RATE"].isna().sum()
    print(f"   Number of NaN flow rates: {nan_count}")

    # Show rows where FLOW_RATE is NaN
    nan_rows = df[df["FLOW_RATE"].isna()]
    print("   Rows with NaN flow rates:")
    # print(" " * 3, nan_rows)
    print(nan_rows)

    # Get all unique quality codes in the dataset
    all_quality_codes = df["QUALITY"].unique()
    print(f"\nAll quality codes present: {sorted(all_quality_codes)}")
    print(
        "   - For quality code descriptions see: https://www.dws.gov.za/Hydrology/Verified/HyCodes.aspx"
    )
    # Count occurrences of each quality code
    quality_counts = df["QUALITY"].value_counts().sort_index()
    print("   - Quality code distribution:")
    print(quality_counts)


def average_data(daily_data_dir: str):

    print(f"Averaging dialy data from `{daily_data_dir}`")
    df = pd.read_csv(daily_data_dir)

    # Ensure DATE is datetime type (if not already)
    df["DATE"] = pd.to_datetime(df["DATE"])

    # 1. Create Month-Year column for grouping
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month

    # 2. Calculate monthly averages (grouping by both Year and Month)
    monthly_avg = (
        df.groupby(["YEAR", "MONTH"])
        .agg(
            {
                "FLOW_RATE": "mean",  # Average flow rate (auto-excludes NaN)
                "QUALITY": lambda x: x.mode()[0],  # Most frequent quality code
            }
        )
        .reset_index()
    )

    # 3. Add month name for readability (optional)
    monthly_avg["MONTH_NAME"] = monthly_avg["MONTH"].apply(
        lambda x: pd.to_datetime(x, format="%m").strftime("%b")
    )

    # 4. Save to CSV
    station_code = os.path.splitext(os.path.basename(daily_data_dir))[0]
    save_path = f"./data/stream_flow_data/monthly/{station_code}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    monthly_avg.to_csv(save_path, index=False)

    print(f"Monthly averages saved to: {save_path}")


def calc_sdi(df, k=3):
    """
    Calculate cumulative Streamflow Drought Index (SDI) for given time scale k (months).

    Parameters:
        df (pd.DataFrame): must have columns ['YEAR','MONTH','FLOW_RATE']
        k (int): time scale in months (e.g. 3, 6, 12)

    Returns:
        pd.DataFrame with added 'SDI_k' column
    """

    # Sort chronologically
    df = df.sort_values(by=["YEAR", "MONTH"]).reset_index(drop=True)

    # Compute cumulative volume over k months
    df[f"V_{k}"] = df["FLOW_RATE"].rolling(window=k, min_periods=k).sum()

    # Compute mean and std for available cumulative volumes
    mean_Vk = df[f"V_{k}"].mean(skipna=True)
    std_Vk = df[f"V_{k}"].std(skipna=True)

    # Standardize
    df[f"SDI_{k}"] = (df[f"V_{k}"] - mean_Vk) / std_Vk

    return df

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

def main():
    # ==================== Pick CSV File ====================
    data_file = "./data/stream_flow_data/monthly/B1H012.csv"

    # ==================== Extract Data ====================
    df = pd.read_csv(data_file)

    # ==================== Calc SDI ====================
    sdi_df = calc_sdi(df)
    print(sdi_df)

    save_dataframe(sdi_df, "./../combine_indices/data/sdi.csv", debug=True)
    return

    # ==================== Applying Filters Example ====================
    conds = [sdi_df["YEAR"] == 2020, sdi_df["MONTH"] == 3]
    print(apply_filters(sdi_df, conds))

    print(sdi_df.head())
    # data_path = "./data/stream_flow_data/daily/B1H012.csv"
    # average_data(data_path)

    return
    df = pd.read_csv(data_path)

    assess_data(df)
    return
    flow_rates = df.loc[:, "FLOW_RATE"]

    print(pd.isna(flow_rates).sum())
    return

    print(df.head())

    print(df.describe())

    # nan_flows = pd.isna(df[df["FLOW_RATE"]])
    # print(nan_flows)

    # scrape_stream_flow("B1H012")
    return


if __name__ == "__main__":
    main()
