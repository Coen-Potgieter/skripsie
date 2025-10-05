import numpy as np
import pandas as pd
import time
import os
import concurrent.futures
import xarray as xr
import requests as re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime


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


def print_heading(heading, num_equals=20):
    print("=" * num_equals, heading, "=" * num_equals)


def visualise_ndvi_region(ndvi_data, title="NDVI Region"):
    """
    Plot the NDVI data on a map using Cartopy
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot the NDVI data
    im = ndvi_data.plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap="YlGn", add_colorbar=True
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.gridlines(draw_labels=True)

    # Set extent to show the region clearly
    ax.set_extent(
        [
            ndvi_data.longitude.min() - 1,
            ndvi_data.longitude.max() + 1,
            ndvi_data.latitude.min() - 1,
            ndvi_data.latitude.max() + 1,
        ],
        crs=ccrs.PlateCarree(),
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def extract_ndvi_by_region(xrds, lat_min, lat_max, lon_min, lon_max):
    """
    Extract NDVI values within a specified latitude and longitude range.

    Parameters:
    xrds (xarray.Dataset): Input dataset containing NDVI variable
    lat_min (float): Minimum latitude (southern boundary)
    lat_max (float): Maximum latitude (northern boundary)
    lon_min (float): Minimum longitude (western boundary)
    lon_max (float): Maximum longitude (eastern boundary)

    Returns:
    xarray.DataArray: NDVI values within the specified region
    """
    # Select the region using slice or where
    region_ndvi = xrds["NDVI"].sel(
        latitude=slice(
            lat_max, lat_min
        ),  # Note: latitude is descending (89.97 to -89.97)
        longitude=slice(lon_min, lon_max),
    )

    return region_ndvi


def check_valid_response(r: re.Response):
    if r.status_code != 200:
        return False
    return True


# Example usage for Western Cape, South Africa:
def extract_western_cape_ndvi(xrds):
    """
    Extract NDVI values for Western Cape province, South Africa.

    Western Cape approximate bounds:
    - Latitude: -35.0° to -31.0° (South to North)
    - Longitude: 17.0° to 23.0° (West to East)
    """
    return extract_ndvi_by_region(
        xrds,
        lat_min=-35.0,  # Southern boundary
        lat_max=-31.0,  # Northern boundary
        lon_min=17.0,  # Western boundary
        lon_max=23.0,  # Eastern boundary
    )


def xarray_basics(xrds: xr.Dataset):
    """
    This is a tutorial of sorts for xarray and `.nc` files :)
    """

    print("\n")

    # Simply printing the thing gives us metadata about the dataset (size, coordinates with cardinality, data variables, then file metadata)
    print("Printing Formatted Meta Data Of the Data Set...\n", xrds)
    print("=" * 40, "\n")

    # Returns a python dictionary
    file_metadata = xrds.attrs

    # Gives back info about coordinates (For us its time, lat and long)
    coord_attrs = xrds.coords
    print("Printing Coord Attributes...\n", coord_attrs)
    print("=" * 40, "\n")

    # Gives info about the data variables, not that these are indexed by our coordinates
    data_variables = xrds.data_vars
    print("Printing Data Variables...\n", data_variables)
    print("=" * 40, "\n")

    # Access specific Data variable (This does not give data values, just info about this variable like name, units, etc.)
    ndvi_variable = xrds.data_vars["NDVI"]
    print("Printing NDVI Data Variable...\n", ndvi_variable)
    print("=" * 40, "\n")

    # This returns a Numpy array, since we have 3 coords, it is 3 dimensional array
    ndvi_data = xrds.data_vars["NDVI"].values
    print("Printing Info of NDVI Numpy Array...")
    print("Shape: ", ndvi_data.shape)
    print("Dim 1 (time) Size:", xrds.coords["time"].values.size)
    print("Dim 2 (latitude) Size:", xrds.coords["latitude"].values.size)
    print("Dim 3 (longitude) Size:", xrds.coords["longitude"].values.size)
    print("=" * 40, "\n")
    return


def understand_input_data(xrds: xr.Dataset):
    """
    Just small helper function to understand the input data
    """
    num_equals = 20
    print("=" * num_equals, "Understanding Data", "=" * num_equals)

    # Ensure we have 1 time variable
    assert xrds.sizes["time"] == 1

    # print this time variable
    datetime = xrds.coords["time"].values[0]
    print("This File Date: ", datetime)

    # print Latitude value ranges
    lat_values = xrds.coords["latitude"].values
    print("Lattitude Info:")
    print(f"    - Values Range From [{np.min(lat_values)}, {lat_values.max()}]")
    print(f"    - Intervals are {lat_values[1] - lat_values[0]}")
    print(f"    - First 5 values {lat_values[0:5]}")

    long_values = xrds.coords["longitude"].values
    print("Longitude Info:")
    print(f"    - Values Range From [{np.min(long_values)}, {long_values.max()}]")
    print(f"    - Intervals are {long_values[1] - long_values[0]}")
    print(f"    - First 5 values {long_values[0:5]}")

    print("HOPE THAT HELPS :)")


def download_single_file(url, output_dir):
    """Download a single file"""
    try:
        filename = os.path.join(output_dir, url.split("/")[-1])
        response = re.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filename
    except Exception as e:
        return f"Error: {url} - {e}"


def download_parallel(url_list, output_dir="~/Downloads", max_workers=5):
    """
    Download files in parallel for faster performance
    """
    os.makedirs(output_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each download
        future_to_url = {
            executor.submit(download_single_file, url, output_dir): url
            for url in url_list
        }

        results = []
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed: {url}")
            except Exception as e:
                print(f"Error with {url}: {e}")

    return results


def scrape_downlaod_list(url):
    response = re.get(url)
    check_valid_response(response)

    html_content = response.content

    soup = BeautifulSoup(html_content, "html.parser")

    links = soup.find_all("a", href=True)

    download_list = []
    for link in links:
        single_link = link["href"]

        if single_link[-3:].strip() != ".nc":
            continue

        download_list.append(url + single_link)

    return download_list


def process_directory(
    input_dir, lat_min=-35.0, lat_max=-22.0, lon_min=16.0, lon_max=33.0
):
    """
    Process all .nc files in a directory by extracting a specific region (South Africa by default)
    and replacing the original file with the processed version.

    Parameters:
    input_dir (str): Directory containing .nc files to process
    lat_min (float): Minimum latitude (southern boundary)
    lat_max (float): Maximum latitude (northern boundary)
    lon_min (float): Minimum longitude (western boundary)
    lon_max (float): Maximum longitude (eastern boundary)
    """
    # Convert to Path object for easier handling
    input_path = Path(input_dir)

    # Find all .nc files in the directory
    nc_files = list(input_path.glob("*.nc"))

    if not nc_files:
        print(f"No .nc files found in {input_dir}")
        return

    print(f"Processing {len(nc_files)} files in {input_dir}...")

    for nc_file in nc_files:
        try:
            print(f"Processing {nc_file.name}...")

            # Open the NetCDF file
            with xr.open_dataset(nc_file, decode_cf=False) as ds:
                # Extract the South Africa region
                # Note: latitude is descending (north to south), so we use slice(lat_max, lat_min)
                sa_ds = ds.sel(
                    latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
                )

                # Close the original dataset
                ds.close()

                # Create a temporary file name
                temp_file = nc_file.with_suffix(".temp.nc")

                # Save the processed data to temporary file
                sa_ds.to_netcdf(temp_file)

                # Close the processed dataset
                sa_ds.close()

            # Remove the original file
            os.remove(nc_file)

            # Rename the temporary file to the original name
            os.rename(temp_file, nc_file)

            print(f"✓ Successfully processed {nc_file.name}")

        except Exception as e:
            print(f"✗ Error processing {nc_file.name}: {e}")
            # Clean up temporary file if it exists
            temp_file = nc_file.with_suffix(".temp.nc")
            if temp_file.exists():
                os.remove(temp_file)


def scrape_all_years_south_africa():
    years = [i for i in range(1981, 2026)]
    for year in years:
        start = time.time()
        print_heading("Scraping Download List")
        download_list = scrape_downlaod_list(
            f"https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/{year}/"
        )
        print_heading("Parallel Downloading")
        download_parallel(download_list, output_dir=f"./data/raw_downloads/{year}")

        print_heading(f"Processing Files For Year: {year}")
        process_directory(f"./data/raw_downloads/{year}")
        end = time.time()
        print_heading(f"✓ Finished With Year: {year} In {end - start}s")


def create_dataset_for_area(coords):
    """ """

    # Get all files in donwlaods
    raw_downlaods_path = "./data/raw_downloads/"

    # Get all the years we are processing
    all_entries = os.listdir(raw_downlaods_path)

    # Filter out only the directories
    years = [
        entry
        for entry in all_entries
        if os.path.isdir(os.path.join(raw_downlaods_path, entry))
    ]

    all_years_df = pd.DataFrame()
    for year in years:
        print(f"Processing {year}...")
        year_path = os.path.join(raw_downlaods_path, year)

        all_files = os.listdir(year_path)

        netcdf_files = [
            entry
            for entry in all_files
            if os.path.isfile(os.path.join(year_path, entry)) and entry[-3:] == ".nc"
        ]

        daily_data = []
        for file in netcdf_files:
            # open .nc
            file_path = os.path.join(year_path, file)
            xrds = xr.open_dataset(file_path, decode_cf=False)

            # Extract region and convert to numpy array
            region_data = extract_ndvi_by_region(xrds, **coords).values.astype(float)

            # Take out fill values of -9999
            region_data[region_data == -9999] = np.nan
            # Rescale valid values
            region_data = region_data / 10000.0

            # Compute mean, ignoring NaNs
            mean_ndvi = np.nanmean(region_data)
            # This will give yyyymmdd
            date = datetime.strptime(file.split("_")[-2].strip(), "%Y%m%d").date()

            daily_data.append(
                {
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "DATE": date,
                    "NDVI": mean_ndvi,
                }
            )
            xrds.close()

        daily_df = pd.DataFrame(daily_data)
        monthly_df = daily_df.groupby(["year", "month"])["NDVI"].mean().reset_index()

        all_years_df = pd.concat([all_years_df, monthly_df], ignore_index=True)
        print("✓ Success\n")

    # sort the final dataframe
    final_df = all_years_df.sort_values(["year", "month"]).reset_index(drop=True)

    # TODO: Maybe think of incorporating more areas
    # save as a csv
    save_dataframe(final_df, "./data/processed_data/ndvi.csv", True)


def validate_ndvi(df: pd.DataFrame) -> None:
    """
    Validate NDVI dataframe with columns: year, month, NDVI.
    Prints validation results to console.
    """
    print("NDVI Data Validation Report")
    print("=" * 35)

    # 1. Column check
    expected_cols = {"year", "month", "NDVI"}
    print(f"Has expected columns: {set(df.columns) >= expected_cols}")

    # 2. Range check
    in_range = df["NDVI"].between(-1, 1).all()
    min = df["NDVI"].min().round(4)
    max = df["NDVI"].max().round(4)
    print(f"NDVI values should be between -1 & 1. They are between: [{min}, {max}]")

    # 3. Temporal order check
    ordered = df.sort_values(["year", "month"]).equals(df)
    print(f"Data ordered by year/month: {ordered}")

    # 4. Mean sanity
    mean_val = df["NDVI"].mean()
    print(f"Mean NDVI: {mean_val:.3f} (reasonable: {-0.2 <= mean_val <= 0.9})")

    # 5. Missing values
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

    # 6. Seasonal pattern quick check
    if "month" in df.columns and not df.empty:
        growing = df[df["month"].between(9, 3)]["NDVI"].mean()
        if not np.isnan(growing):
            print(f"Seasonal check passed (mean in growing season: {growing:.3f})")
        else:
            print("Seasonal check skipped (no data)")
    else:
        print("Seasonal check skipped (no month column or empty data)")

    print("=" * 35)


def main():

    # ==================== Check Data Quality ====================
    data = pd.read_csv("./data/processed_data/ndvi.csv")
    validate_ndvi(data)

    print(data.head())
    return

    # ==================== Define Study Area Coords ====================
    study_area_coords = {
        "lat_max": -30.7,
        "lat_min": -34.83,
        "lon_min": 17.85,
        "lon_max": 21.17,
    }

    # ==================== Create CSV Data Set For lat-long Area====================
    create_dataset_for_area(study_area_coords)
    return

    # ==================== Visulise Specific Region ====================

    dataset = "./data/raw_downloads/1981/AVHRR-Land_v005_AVH13C1_NOAA-07_19810624_c20170610041337.nc"
    # Load your dataset
    xrds = xr.open_dataset(dataset, decode_cf=False)

    # # Extract Western Cape data
    # wc_data = extract_western_cape_ndvi(xrds)
    # visualise_ndvi_region(wc_data, "NDVI - Western Cape")

    study_area_coords = {
        "lat_max": -30.7,
        "lat_min": -34.83,
        "lon_min": 17.85,
        "lon_max": 21.17,
    }
    cape_town_data = extract_ndvi_by_region(xrds, **study_area_coords)
    print(cape_town_data)

    # return
    visualise_ndvi_region(cape_town_data, "NDVI - Western Cape")

    # Or use the flexible function with custom bounds

    # print(f"Western Cape NDVI shape: {western_cape_ndvi.shape}")
    # print(f"Western Cape NDVI data:\n{western_cape_ndvi}")

    # visualise_ndvi_region(xrds.data_vars["NDVI"], "NDVI - Western Cape")
    return

    xrds = xr.open_dataset("./data/test.nc", decode_cf=False)
    # xrds = xr.open_dataset("./data/test.nc")
    # return
    # understand_input_data(xrds)
    xarray_basics(xrds)

    # understand_input_data(xrds)
    # playing_some(xrds)


if __name__ == "__main__":
    main()
