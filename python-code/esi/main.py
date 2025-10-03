import numpy as np
import xarray as xr
from datetime import datetime


def xarray_basics(xrds: xr.Dataset):
    '''
    This is a tutorial of sorts for xarray and `.nc` files :)
    '''

    # Simply printing the thing gives us metadata about the dataset (size, coordinates with cardinality, data variables, then file metadata)
    # print(xrds)
    
    # Returns a python dictionary
    file_metadata = xrds.attrs


    # Gives back info about coordinates (For us its time, lat and long)
    coord_attrs = xrds.coords

    # Gives info about the data variables, not that these are indexed by our coordinates
    data_variables = xrds.data_vars

    # Access specific Data variable (This does not give data values, just info about this variable like name, units, etc.)
    et_variable = xrds.data_vars['ETos']

    # This returns a Numpy array, since we have 3 coords, it is 3 dimensional array
    et_data = xrds.data_vars['ETos'].values

    assert(type(et_data) == np.ndarray) # Ensure it is numpy array
    assert(len(et_data.shape) == 3) # Ensure it is 3 dimensional

    print(et_data)

    

def understand_input_data(xrds: xr.Dataset):
    '''
    Just small helper function to understand the input data
    '''
    num_equals = 20
    print("="*num_equals, "Understanding Data", "="*num_equals)

    # Ensure we have 1 time variable
    assert(xrds.sizes['time'] == 1)

    # print this time variable
    datetime = xrds.coords['time'].values[0]
    print("This File Date: ", datetime)

    # print Lattitude value ranges
    lat_values = xrds.coords['lat'].values
    print("Lattitude Info:")
    print(f"    - Values Range From [{np.min(lat_values)}, {lat_values.max()}]")
    print(f"    - Intervals are {lat_values[1] - lat_values[0]}")
    print(f"    - First 5 values {lat_values[0:5]}")

    long_values = xrds.coords['lon'].values
    print("Longitude Info:")
    print(f"    - Values Range From [{np.min(long_values)}, {long_values.max()}]")
    print(f"    - Intervals are {long_values[1] - long_values[0]}")
    print(f"    - First 5 values {long_values[0:5]}")


    print("HOPE THAT HELPS :)")


def playing_some(xrds: xr.Dataset):

    # 3-dim (time, lat, lon), assertions below just make sure of this...
    et_values = xrds.data_vars['ETos'].values

    # Just assertions to ensure that our values are indexxed (time, lat, lon)
    attr_names = ["time", "lat", "lon"]
    for i in range(len(attr_names)):
        assert(et_values.shape[i] == xrds.sizes[attr_names[i]])

    max_lat = -22
    max_long = 33
    min_lat = -35
    min_long = 16.3

    lat_values = xrds.coords['lat'].values

    # found_idx = 

    print(lat_values)


    print(et_values[0, 0, 0])




    pass

def main():

    xrds = xr.open_dataset("./data/2003/ETos_20030101.nc")
    # xarray_basics(xrds)
    understand_input_data(xrds)
    # playing_some(xrds)



if __name__ == "__main__":
    main()

