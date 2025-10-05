import warnings
import folium

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
    Save DataFrame to a CSV file, creating directories if needed.

    Args:
        df: DataFrame to save
        csv_path: Path where CSV file will be saved
        debug: Whether to print debug information

    Returns:
        None
    """

    # Extract directory path from the full file path
    directory = os.path.dirname(csv_path)

    # Create directories if they don't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        if debug:
            print(f"Created directory: `{directory}`")

    # Save the DataFrame
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
        "H",
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


def scrape_all_breede_olifant_stations():
    """
    one of the endpoints broke, the study area is in Breede-Olifants.
    I found this other end point: https://www.dws.gov.za/Hydrology/Unverified/, which lists the stations in that area
    Unfortunately it is an interactive map, ie. Cannot get what I want using endpoints.
    I could use selenium to navigate but Im just gonna copy the html since I only need this once...
    """
    with open("./data/stations_html.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    # Create Beautiful Soup object
    soup = BeautifulSoup(html_content, "html.parser")

    table_rows = soup.find_all("tr")
    # Remove header
    table_rows.pop(0)

    station_codes = []
    for row in table_rows:
        if row["bgcolor"] != "White":
            continue
        row_station_code = row.find("td").find("a").text

        # We now filter out all non-River stations
        #   - These are stations that don't have an `H` as the second letter

        if row_station_code[2] != "H":
            continue

        station_codes.append(row_station_code)

    # Now have all River type stations in Breede-Olifants
    # We will now check if we have meta data on this station, if not then save it

    for code in station_codes:
        area_code = check_valid_region(code[0])

        meta_dir = f"./data/station_data/river/region_{area_code}.json"
        # Open particular meta_data_json
        with open(meta_dir, "r") as tf:
            meta_data = json.load(tf)

        already_saved_codes = [elem["code"] for elem in meta_data]
        if code in already_saved_codes:
            print(code, "Already Saved")
            continue
        else:
            print(code, "Not Saved, Saving Now...")

        # Now we scrape the station and append the json
        target_link = (
            f"https://www.dws.gov.za/hydrology/Verified/HyDataSets.aspx?Station={code}"
        )

        r = re.get(target_link)
        check_valid_response(r)

        html_content = r.content
        soup = BeautifulSoup(html_content, "html.parser")
        lat = None
        try:
            lat = soup.find(id="tbLat")["value"]
        except:
            print(f"\t{code} Is Invalid")
            continue

        lon = soup.find(id="tbLong")["value"]

        start_data = soup.find("input", {"name": "ctl06"})["value"]
        end_data = soup.find(id="tbEnd_0")["value"]
        station_name = soup.find(id="labPlace").text
        new_entry = {
            "code": code,
            "name": station_name,
            "catchment_area": "FuckYou",
            "lat": lat,
            "long": lon,
            "data_avail": f"{start_data} to {end_data}",
        }

        meta_data.append(new_entry)

        print("\t- Saving New Json")

        with open(meta_dir, "w") as f:
            json.dump(meta_data, f, indent=4)


def scrape_station_metadata(region_code: str):
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

    save_dir = f"./data/station_data/river/region_{region_code}.json"
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


def scrape_many_stations(station_codes, overwrite=False):

    for station in station_codes:
        if not overwrite:
            # Check if it exists already, then dont scrape
            entries = os.listdir("./data/stream_flow_data/daily/")
            saved_stations = [
                elem.split(".")[0].strip() for elem in entries if elem[-4:] == ".csv"
            ]

            if station in saved_stations:
                print(station, "Already Saved, Skipping...")
                continue

        faulty_stations = ["G2H013"]
        if station in faulty_stations:
            continue

        print("Scraping Data For", station)
        scrape_stream_flow(station)


def scrape_stream_flow(station_code: str, start_date=None, end_date=None, silent=False):
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

    if not silent:
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
            if not silent:
                print(
                    f"Available Data begins @ {available_data_start.strftime('%Y-%m-%d')}, which is after given date: {start_date.strftime('%Y-%m-%d')}. Therefore getting data from {available_data_start.strftime('%Y-%m-%d')} onwards"
                )
            start_date = available_data_start
    else:
        if not silent:
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
        f"✓ Successfully saved stream flow data for station {station_code} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} @ {save_dir}"
    )


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
    df[f"SDI"] = (df[f"V_{k}"] - mean_Vk) / std_Vk

    # rename
    df.rename(columns={"YEAR": "year", "MONTH": "month"}, inplace=True)

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


def scrape_area(region_code):
    # Validate the code
    region_code = check_valid_region(region_code)

    # Extract all stations

    with open(f"./data/station_data/river/region_{region_code}.json", "r") as file:
        station_data = json.load(file)

    directory_path = "./data/stream_flow_data/daily/"

    # Get all entries (files and directories)
    all_entries = os.listdir(directory_path)

    # Filter for only files
    files_only = [
        entry
        for entry in all_entries
        if os.path.isfile(os.path.join(directory_path, entry))
    ]

    faulty_stations = ["G2H013"]
    new_faulty_stations = []

    csv_files = [item[:-4] for item in files_only]
    for station in station_data:
        # print(station["code"])
        station_code = station["code"]
        if station_code in csv_files:
            continue

        if station_code in faulty_stations:
            continue

        run = True
        counter = 0
        while run:
            try:
                scrape_stream_flow(station_code, silent=True)
            except Exception as e:
                counter += 1
                print(f"{station_code} IS FAULTY FOR THE {counter} TIME....")
                if counter >= 10:
                    new_faulty_stations.append(station_code)
                    run = False
            else:
                run = False
        print()

    if len(new_faulty_stations) > 0:
        print("New Faulty Stations:")
        for station in new_faulty_stations:
            print(station)


def messing_around():
    entries = os.listdir("./data/stream_flow_data/daily/")
    csv_files = [elem for elem in entries if elem.strip()[-4:] == ".csv"]

    area_code_dict = {}
    station_meta_data_path = "./data/station_data/river/"

    for file in csv_files:
        area_code = check_valid_region(file[0])

        print(area_code)
    pass


def get_stations_by_study_area(area_coords):

    # First Extract All Station Meta Data Into One Big List
    station_meta = []
    meta_dir = "./data/station_data/river/"
    all_entries = os.listdir(meta_dir)
    json_files = [elem for elem in all_entries if elem[-5:] == ".json"]

    for file in json_files:
        with open(os.path.join(meta_dir, file), "r") as tf:
            data = json.load(tf)

        for elem in data:
            station_meta.append(elem)

    station_codes_inside = []
    for station in station_meta:

        inside = (
            (float(station["lat"]) >= area_coords["lat_min"])
            & (float(station["lat"]) <= area_coords["lat_max"])
            & (float(station["long"]) >= area_coords["lon_min"])
            & (float(station["long"]) <= area_coords["lon_max"])
        )

        if inside:
            station_codes_inside.append(station["code"])

    return station_codes_inside


def plot_stations_by_codes(codes):
    # First Extract All Station Meta Data Into One Big List
    station_meta = []
    meta_dir = "./data/station_data/river/"
    all_entries = os.listdir(meta_dir)
    json_files = [elem for elem in all_entries if elem[-5:] == ".json"]

    for file in json_files:
        with open(os.path.join(meta_dir, file), "r") as tf:
            data = json.load(tf)

        for elem in data:
            station_meta.append(elem)

    target_meta_data = []
    for code in codes:
        for station in station_meta:
            if code == station["code"]:
                target_meta_data.append(
                    {
                        "code": code,
                        "lat": station["lat"],
                        "lon": station["long"],
                    }
                )

    df = pd.DataFrame(target_meta_data)

    # Must convert to numeric columns...
    df["lat"] = pd.to_numeric(df["lat"])
    df["lon"] = pd.to_numeric(df["lon"])

    # Create a base map centered on the mean coordinates
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    # Add markers for each station
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=row["code"],
            tooltip=row["code"],
        ).add_to(m)

    # Save the map
    save_file = "area_of_study_map.html"
    m.save(save_file)
    print(f"Interactive map saved as {save_file}")

    pass


def avg_data_many_stations(station_codes):

    daily_dir = "./data/stream_flow_data/daily/"
    entries = os.listdir(daily_dir)
    csv_files = [elem for elem in entries if elem[-4:] == ".csv"]

    for station in station_codes:
        for file in csv_files:
            if station == file.split(".")[0].strip():
                average_data(os.path.join(daily_dir, file))


def calc_sdi_many_stations(station_codes):
    faulty_stations = ["G2H013"]
    list_of_dfs = []

    for station in station_codes:

        if station in faulty_stations:
            continue
        target_file = f"./data/stream_flow_data/monthly/{station}.csv"
        # Extract Data
        df = pd.read_csv(target_file)

        # Calc SDI
        list_of_dfs.append(calc_sdi(df))

    big_sdi_df = pd.concat(list_of_dfs)

    # Assuming your dataframe is named 'df'
    # Group by year and month, calculate mean SPI, and rename to SDI
    result_df = big_sdi_df.groupby(["year", "month"], as_index=False)["SDI"].mean()

    # Keep only the required columns
    result_df = result_df[["year", "month", "SDI"]]

    # Dropping first block of NaN values
    # 1. Find the index of the first row that does NOT have a NaN in the 'SDI' column
    first_valid_row_index = result_df["SDI"].first_valid_index()

    # 2. Slice the DataFrame from that index to the end
    #    This automatically drops all preceding NaN records.
    result_df = result_df.loc[first_valid_row_index:]

    print(result_df.head(20))
    print(f"\nShape: {result_df.shape}")
    print(f"\nTotal rows: {len(result_df)}")
    return result_df


def validate_sdi(df: pd.DataFrame) -> None:
    """
    Validate SDI dataframe with columns: year, month, SDI.
    Prints validation results to console.
    """
    print("SDI Data Validation Report")
    print("=" * 35)

    # 1. Column check
    expected_cols = {"year", "month", "SDI"}
    print(f"Has expected columns: {set(df.columns) >= expected_cols}")

    # 2. Range check (SDI typically ranges from -3 to +3, but can exceed)
    min_val = df["SDI"].min()
    max_val = df["SDI"].max()
    print(
        f"SDI values typically between -3 & 3. They are between: [{min_val:.4f}, {max_val:.4f}]"
    )
    extreme_count = ((df["SDI"] < -3) | (df["SDI"] > 3)).sum()
    print(
        f"Values outside [-3, 3] range: {extreme_count} ({extreme_count/len(df)*100:.1f}%)"
    )

    # 3. Temporal order check
    ordered = df.sort_values(["year", "month"]).equals(df)
    print(f"Data ordered by year/month: {ordered}")

    # 4. Mean sanity (SDI should have mean close to 0)
    mean_val = df["SDI"].mean()
    print(f"Mean SDI: {mean_val:.3f} (should be close to 0: {-0.5 <= mean_val <= 0.5})")

    # 5. Standard deviation check (SDI should have std close to 1)
    std_val = df["SDI"].std()
    print(f"Std Dev SDI: {std_val:.3f} (should be close to 1: {0.8 <= std_val <= 1.2})")

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
    if not df.empty and not df["SDI"].isna().all():
        drought_severe = (df["SDI"] <= -1.5).sum()
        drought_moderate = ((df["SDI"] > -1.5) & (df["SDI"] <= -1.0)).sum()
        normal = ((df["SDI"] > -1.0) & (df["SDI"] < 1.0)).sum()
        wet_moderate = ((df["SDI"] >= 1.0) & (df["SDI"] < 1.5)).sum()
        wet_severe = (df["SDI"] >= 1.5).sum()

        total = len(df)
        print(f"Condition distribution:")
        print(
            f"\tSevere drought (SDI ≤ -1.5): {drought_severe} ({drought_severe/total*100:.1f}%)"
        )
        print(
            f"\tModerate drought (-1.5 < SDI ≤ -1.0): {drought_moderate} ({drought_moderate/total*100:.1f}%)"
        )
        print(f"\tNormal (-1.0 < SDI < 1.0): {normal} ({normal/total*100:.1f}%)")
        print(
            f"\tModerately wet (1.0 ≤ SDI < 1.5): {wet_moderate} ({wet_moderate/total*100:.1f}%)"
        )
        print(f"\tSeverely wet (SDI ≥ 1.5): {wet_severe} ({wet_severe/total*100:.1f}%)")
    else:
        print("Condition distribution skipped (no valid data)")

    print("=" * 35)


def main():

    study_area_coords = {
        "lat_max": -30.7,
        "lat_min": -34.83,
        "lon_min": 17.85,
        "lon_max": 21.17,
    }

    station_codes = get_stations_by_study_area(study_area_coords)

    # return
    # plot_stations_by_codes(station_codes)
    # scrape_many_stations(station_codes)

    # avg_data_many_stations(station_codes)

    final = calc_sdi_many_stations(station_codes)
    validate_sdi(final)

    # Save Data
    save_dataframe(final, "./data/sdi_data/sdi.csv", debug=True)

    return

    # scrape_all_breede_olifant_stations()
    # messing_around()
    # return

    # data = pd.read_csv("./data/stream_flow_data/monthly/B1H012.csv")
    # print(data.head())
    # return
    # scrape_station_metadata("H")

    # ==================== Plot Stations In Area ====================
    # areas = ["A", "H", "V"]
    # plot_stations(areas)
    # return

    # ==================== Scrape All Stations In Area ====================
    # scrape_area("K")
    # return

    # ==================== Scrape Station ====================

    scrape_many_stations(["G2H040"])
    # scrape_stream_flow("G2H040")
    return

    # ==================== Convert Daily Streamflow To Monthly Streamflow ====================
    # average_data("./data/stream_flow_data/daily/B1H012.csv")
    # return

    # ==================== Calculate SDI From Monthly Data ====================
    # Pick CSV File
    data_file = "./data/stream_flow_data/monthly/B1H012.csv"

    # Extract Data
    df = pd.read_csv(data_file)

    # Calc SDI
    sdi_df = calc_sdi(df)

    # Save Data
    save_dataframe(sdi_df, "./data/sdi_data/sdi.csv", debug=True)
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
