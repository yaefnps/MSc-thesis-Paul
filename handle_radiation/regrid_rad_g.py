import os
import sys
import calendar
import datetime as dt
import numpy as np
import xarray as xr

# regridding handling shared with the rest of the project
from era5_to_flex_helpers import (
    read_hybrid_coefficients,
    read_met,
    calculate_geometry,
    average_to_flexpart_layers,
    wrap_lon,
    as_array,
)
# radiation file handling + date list stay in the old helper
from regrid_to_z_levels_newnaming import open_rad_day, create_datelist

## directories ##
eradir_fx = '/jetfs/shared-data/ECMWF/ERA5_glob_0.5deg_1h/'
eradir_rad = '/jetfs/scratch/psplechtna/thesisdata/radfluxes/hybrid/'
eradir_out = '/jetfs/scratch/psplechtna/thesisdata/radfluxes/z/'

## full month from command line: python regrid_rad_g.py YEAR MONTH ##
if len(sys.argv) != 3:
    sys.exit('Usage: python regrid_rad_g.py YEAR MONTH')
year, month = int(sys.argv[1]), int(sys.argv[2])
n_days = 2
#n_days = calendar.monthrange(year, month)[1]
start_time = dt.datetime(year, month, 1, 0)
times = create_datelist(start_time, start_time + dt.timedelta(days=n_days, hours=-1))
daylist_era = np.unique(np.array([dt.datetime(t.year, t.month, t.day) for t in times]))

## target layers: [h0,h1], [h1,h2], ... (average_to_flexpart_layers starts at 0) ##
heightlist = [0, 20, 50, 100, 200, 300, 400, 500,
              600, 700, 800, 900, 1000, 1100, 1200, 1300,
              1400, 1500, 1600, 1700, 1800, 1900, 2000,
              2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600,
              3800, 4000, 4200, 4400, 4600, 4800, 5000, 5500,
              6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,
              11000, 12000, 15000, 20000]
heights = np.array(heightlist, dtype=float)
upper_bounds = heights[1:]
var_list = ['avg_ttswr', 'avg_ttlwr']


def fx_file(t):
    return eradir_fx + t.strftime('%Y/%m/EA%Y%m%d%H')


## grid + hybrid coefficients, taken once from the first fx file ##
first_file = fx_file(times[0])
a_half, b_half = read_hybrid_coefficients(first_file)

with xr.open_dataset(first_file, engine='cfgrib',
                     backend_kwargs={'filter_by_keys': {'paramId': 130}, 'indexpath': ''}) as ds0:
    ds0 = ds0.assign_coords(longitude=wrap_lon(ds0.longitude)).sortby('longitude')
    latitude, longitude = ds0.latitude.values, ds0.longitude.values


def process_day(d):
    """Regrid one day of radiation tendencies; returns {var: (24, nheight-1, nlat, nlon)}."""
    hours = create_datelist(d, d + dt.timedelta(hours=23))

    # bring radiation data onto the same (wrapped, sorted) grid as the fx fields
    rad = open_rad_day(eradir_rad, d, var_list)
    rad = rad.assign_coords(longitude=wrap_lon(rad.longitude)).sortby('longitude')
    rad = rad.sortby('hybrid').reindex(latitude=latitude, longitude=longitude,
                                       method='nearest', tolerance=1e-3)

    out = {var: [] for var in var_list}
    for t in hours:
        met = read_met(fx_file(t), latitude, longitude)
        _, height_half, _ = calculate_geometry(met, a_half, b_half)
        rad_t = rad.sel(valid_time=np.datetime64(t))
        for var in var_list:
            src = as_array(rad_t[var], ('hybrid', 'latitude', 'longitude'))
            out[var].append(average_to_flexpart_layers(src, height_half, upper_bounds).astype(np.float32))

    return {var: np.stack(out[var]) for var in var_list}, [np.datetime64(t) for t in hours]


## read, postprocess and regrid one day at a time ##
var_int_dict = {var: [] for var in var_list}
time_list = []
for d in daylist_era:
    print(d, 'started')
    result, day_times = process_day(d)
    for var in var_list:
        var_int_dict[var].append(result[var])
    time_list.extend(day_times)
    print(d, 'done')

ds_era_int = xr.Dataset(
    data_vars={
        var: (['valid_time', 'height', 'latitude', 'longitude'], np.concatenate(var_int_dict[var]))
        for var in var_list
    },
    coords={
        'longitude': ('longitude', longitude),
        'latitude': ('latitude', latitude),
        'height': ('height', upper_bounds),
        'valid_time': ('valid_time', time_list),
    },
    attrs={'description': 'Height-weighted layer-mean variables (era5_to_flex_helpers)'},
)

os.makedirs(f'{eradir_out}/{year}', exist_ok=True)
ds_era_int.to_netcdf(f'{eradir_out}/{year}/heightav_{year}{month:02d}.nc')
