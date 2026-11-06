import numpy as np
import xarray as xr
import datetime as dt
import datetime
import glob
from regrid_to_z_levels_newnaming import *

g = 9.80655
## directories ##
eradir_fx = '/jetfs/shared-data/ECMWF/ERA5_glob_0.5deg_1h/'
eradir_rad = '/jetfs/scratch/psplechtna/thesisdata/radfluxes/hybrid/'
eradir_out = '/jetfs/scratch/psplechtna/thesisdata/radfluxes/z/'

## create time list for when to regrid ##
year = 2024
month = 6
day = 1
hour = 0

start_time = dt.datetime(year, month, day, hour)
window_days = 30

times = create_datelist(start_time, start_time+dt.timedelta(days=window_days, hours=-1))

## define heights, variables, days ##
heightlist =  [0, 20, 50, 100, 200, 300, 400, 500,
                    600, 700, 800, 900, 1000, 1100, 1200, 1300,
                    1400, 1500, 1600, 1700, 1800, 1900, 2000,
                    2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600,
                    3800, 4000, 4200, 4400, 4600, 4800, 5000, 5500,
                    6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,
                    11000, 12000, 15000, 20000]

daylist_era = np.unique(np.array([dt.datetime(date.year, date.month, date.day) for date in times]))
var_list = ['avg_ttswr', 'avg_ttlwr']
heights = np.array(heightlist)

## read, postprocess and regrid one day at a time ##
var_int_dict = {var: [] for var in var_list}
time_list = []
coords = None
for day in daylist_era:
    print(day, 'started')
    result, coords = process_day(eradir_fx, eradir_rad, day, var_list, heights, g)
    for var in var_list:
        var_int_dict[var].append(result[var])
    time_list.extend(coords['valid_time'])
    print(day, 'done')

ds_era_int = xr.Dataset(
    data_vars={
        var: (['valid_time', 'height', 'latitude', 'longitude'], np.concatenate(var_int_dict[var]))
        for var in var_list
    },
    coords={
        'longitude': ('longitude', coords['longitude']),
        'latitude': ('latitude', coords['latitude']),
        'height': ('height', heights[1:]),
        'valid_time': ('valid_time', time_list),
    },
    attrs={'description': 'Weighted mean variables'},
)

ds_era_int.to_netcdf(f'{eradir_out}/{year}/higherres_'+start_time.strftime('%Y%m%d')+f'_+{window_days}d.nc')