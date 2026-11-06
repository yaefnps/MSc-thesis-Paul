import numpy as np
import xarray as xr
import datetime as dt
import datetime
import glob
import os

def _parse_rad_filename(path):
    """TMTTRAD_2024060107_2024060118.nc -> (start, end), both inclusive."""
    _, s, e = os.path.basename(path).replace('.nc', '').split('_')
    return dt.datetime.strptime(s, '%Y%m%d%H'), dt.datetime.strptime(e, '%Y%m%d%H')


def open_rad_day(eradir_rad, day, var_list):
    """
    Open all radiation files overlapping `day` and return its 24 hours in memory.
    """
    day_start = dt.datetime(day.year, day.month, day.day, 0)
    day_end = day_start + dt.timedelta(hours=23)

    # works for a flat directory or a YYYY/MM subfolder layout
    candidates = (glob.glob(os.path.join(eradir_rad, 'TMTTRAD_*_*.nc')) +
                  glob.glob(os.path.join(eradir_rad, '*', '*', 'TMTTRAD_*_*.nc')))
    files = sorted(f for f in candidates
                   if (lambda se: se[0] <= day_end and se[1] >= day_start)(_parse_rad_filename(f)))
    if not files:
        raise FileNotFoundError(f'No radiation files cover {day:%Y-%m-%d} in {eradir_rad}')

    ds = xr.open_mfdataset(files, combine='nested', concat_dim='valid_time',
                           data_vars='minimal', coords='minimal', compat='override')
    ds = ds.sortby('valid_time')
    _, idx = np.unique(ds['valid_time'].values, return_index=True)   # drop duplicate stamps
    ds = ds.isel(valid_time=idx)
    ds = ds.sel(valid_time=slice(day_start, day_end))

    # keep only what we need; drop GRIB-style leftovers that could clash in the merge
    ds = ds[var_list].drop_vars(['time', 'step'], errors='ignore')

    # make sure the level dimension is called 'hybrid' like in the fx files
    for cand in ('level', 'model_level', 'lev', 'nhyb'):
        if cand in ds.dims and 'hybrid' not in ds.dims:
            ds = ds.rename({cand: 'hybrid'})

    if ds.sizes['valid_time'] != 24:
        print(f'WARNING: {day:%Y-%m-%d} has {ds.sizes["valid_time"]} radiation timesteps, expected 24')
    return ds.load()

def create_datelist(startdate, stopdate, step=datetime.timedelta(hours=1)):
    """
    Constructs a list of dates.

    Parameters
    ----------
    startdate : datetime.datetime
        First date
    stopdate : datetime.datetime
        Last date
    step : datetime.timedelta
        Difference between dates

    Returns
    -------
    datelist : list
        List of dates between startdate and stopdate
    """

    datelist = []
    date = startdate
    while date <= stopdate:
        datelist.append(date)
        date += step

    return datelist


def vertint(z, var, dp, heights, levax=1):
    """
    Compute layer-mean values within predefined height layers.
    Levels near the boundary of two layers are distributed proportionally
    according to pressure difference.

    Parameters
    ----------
    z : numpy.ndarray
        Heights of model levels.
    var : numpy.ndarray
        Variable to project onto height layers.
    dp : numpy.ndarray
        Layer weights (pressure difference).
    heights : array-like
        Target layer interface heights.
    levax : int, optional
        Axis index corresponding to the vertical level dimension.

    Returns
    -------
    numpy.ndarray
        Layer-averaged variable on the target vertical grid.
    """
    #TODO: turn z array around if height increases with level
    # Bring level axis to front
    z_l = np.moveaxis(z, levax, 0)
    var_l = np.moveaxis(var, levax, 0)
    dp_l = np.moveaxis(dp, levax, 0)

    # Make arrays 2D
    nlev = z_l.shape[0]
    nothers = int(np.prod(z_l.shape[1:])) if z_l.ndim > 1 else 1
    z_resh = z_l.reshape((nlev, nothers))
    var_resh = var_l.reshape((nlev, nothers))
    dp_resh = dp_l.reshape((nlev, nothers))

    # Find the layer the model level is in
    ind = np.digitize(z_resh, heights) - 1

    # Initialize output arrays
    values_grid = np.zeros((len(heights), nothers))
    weights_grid = np.zeros((len(heights), nothers))

    ind_other = np.arange(nothers)

    for i in range(nlev):
        # Compute distance to next level and next outheight (up and down)
        ind_upper = np.minimum(len(heights)-1, ind[i]+1)
        ind_lower = np.maximum(0, ind[i]-1)
        if i == nlev-1:
            dist_zlower = z_resh[i] - 0.
            dist_hlower = z_resh[i] - 0.
        else:
            dist_zlower = z_resh[i] - z_resh[i+1]
            dist_hlower = z_resh[i] - heights[ind[i]]
        if i == 0:
            dist_zupper = np.finfo(float).eps
            dist_hupper = np.finfo(float).eps
        else:
            dist_zupper = z_resh[i] - z_resh[i-1]
            dist_hupper = z_resh[i] - heights[ind_upper]

        # "Distribute" the level proportionally to the different layers
        fact_upper = np.maximum(-dist_zupper/2 + dist_hupper,0)/(dist_zlower/2 - dist_zupper/2)
        fact_lower = np.maximum(dist_zlower/2 - dist_hlower,0)/(dist_zlower/2 - dist_zupper/2)
        fact = 1 - fact_lower - fact_upper

        # Add values to grid.
        # NOTE: np.add.at was replaced with plain indexed += here. np.add.at is only
        # needed when the SAME (row, col) index pair can appear twice within a single
        # call. Here ind_other = np.arange(nothers) is always unique within one call,
        # so (ind[i][j], ind_other[j]) pairs can never collide with each other, and
        # plain fancy-index += is safe -- it just also happens to be vectorized and
        # much faster (np.add.at disables numpy's normal buffering to guard against
        # duplicate indices, which is unnecessary overhead here).
        contrib = var_resh[i] * dp_resh[i]

        values_grid[ind[i], ind_other] += fact * contrib
        values_grid[ind_upper, ind_other] += fact_upper * contrib
        values_grid[ind_lower, ind_other] += fact_lower * contrib
        weights_grid[ind[i], ind_other] += fact * dp_resh[i]
        weights_grid[ind_upper, ind_other] += fact_upper * dp_resh[i]
        weights_grid[ind_lower, ind_other] += fact_lower * dp_resh[i]

    var_grid_resh = values_grid[:len(heights)-1] / weights_grid[:len(heights)-1] # uppermost level is "out of domain"
    var_grid_l = var_grid_resh.reshape((var_grid_resh.shape[0], *z_l.shape[1:]))
    var_grid = np.moveaxis(var_grid_l, 0, levax)

    return var_grid


def postprocess_era(eradir_fx, ds_rad_day, date):
    """
    Read and postprocess ERA fields on model and surface levels.

    Parameters
    ----------
    eradir_fx : str
        Directory containing ERA files organized by date.
    eradir_rad: str
        Directory containing ERA files organized by date for radiation values.
    date : datetime.datetime
        Datetime identifying the ERA file to process.

    Returns
    -------
    xarray.Dataset
        Dataset containing geopotential (z), specific humidity (q), temperature (t), 
        pressure difference (dp), and surface geopotential (z_sfc), 
        time-mean tendencies from radiation parametrizations (ttswr, ttlwr).
    """
    eraname_fx = eradir_fx+date.strftime('%Y/%m/EA%Y%m%d%H')

    ds_era_ml = xr.open_dataset(
        eraname_fx, engine='cfgrib',
        backend_kwargs={'indexpath': '', 'read_keys': ['pv'],},
        filter_by_keys={'typeOfLevel': 'hybrid'}
    )

    # radiation for this hour (scalar valid_time, same layout as the fx fields)
    ds_era_rad = ds_rad_day.sel(valid_time=np.datetime64(date))
    ds_era_rad_sw = ds_era_rad[['avg_ttswr']]
    ds_era_rad_lw = ds_era_rad[['avg_ttlwr']]
    
    ds_era_sf1_t2m = xr.open_dataset(
        eraname_fx, engine='cfgrib',
        backend_kwargs={'indexpath': ''},
        filter_by_keys={'typeOfLevel': 'surface', 'dataType': 'an', 'edition': 1, 'shortName' : '2t'}
    )
    ds_era_sf1_z   = xr.open_dataset(
        eraname_fx, engine='cfgrib',
        backend_kwargs={'indexpath': ''},
        filter_by_keys={'typeOfLevel': 'surface', 'dataType': 'an', 'edition': 1, 'shortName' : 'z'}
    )

    ds_era_sf2 = xr.open_dataset(
        eraname_fx, engine='cfgrib',
        backend_kwargs={'indexpath': ''},
        filter_by_keys={'typeOfLevel': 'surface', 'dataType': 'an', 'edition': 2}
    )


    ds_era_sf1_z = ds_era_sf1_z.rename({'z' : 'z_sfc'})
    ds_era = xr.merge((ds_era_ml, ds_era_sf1_t2m, ds_era_sf1_z, ds_era_sf2, ds_era_rad_sw, ds_era_rad_lw), 
                      compat='override', join='inner')
    ds_era = ds_era.drop_vars('time')

    pv = ds_era_ml['t'].attrs['GRIB_pv']
    n = len(pv) // 2
    hyai = pv[:n]
    hybi = pv[n:2 * n]

    p_half = hyai[:,np.newaxis,np.newaxis] + hybi[:,np.newaxis,np.newaxis] * ds_era['sp'].values[np.newaxis,:,:]

    ds_era['p_half'] = (
        ('hybridi', 'latitude', 'longitude'),
        p_half
    )
    ds_era['dp'] = (
        ds_era['t'].dims,
        np.diff(p_half, axis=0)
    )

    ds_era = calc_geopotential(ds_era)

    return ds_era[['z', 'q', 't', 'dp', 'z_sfc', 'avg_ttswr', 'avg_ttlwr']]


def calc_geopotential(ds_era):
    """
    Compute geopotential on full model levels. 
    Adapted from ECMWF script compute_geopotential_on_ml.py
    Gives the same result as the metview function. :)

    Parameters
    ----------
    ds_era : xarray.Dataset
        ERA dataset containing temperature (t), specific humidity (q), pressure (p_half),
        and surface geopotential (z_sfc).

    Returns
    -------
    xarray.Dataset
        Input dataset with an added z variable containing geopotential on
        full model levels.
    """
    R_d = 287.06

    nlev = len(ds_era['hybrid'])
    levax = ds_era['t'].get_axis_num('hybrid')

    # NOTE: pulled everything into plain numpy arrays before the per-level loop.
    # The original loop used ds_era['t'].isel(...), elementwise ops on DataArrays,
    # etc. -- each of those carries xarray overhead (index alignment, attribute
    # bookkeeping, object creation) that adds up badly over 137 levels, and this
    # function runs once per hourly file. Plain numpy inside the loop is the same
    # math, much less overhead.
    t = np.moveaxis(ds_era['t'].values, levax, 0)
    q = np.moveaxis(ds_era['q'].values, levax, 0)
    p_half = ds_era['p_half'].values  # dims are already (hybridi, latitude, longitude)
    z_h = ds_era['z_sfc'].values.copy()

    z_array = np.zeros_like(t)

    for k in range(nlev-1, -1, -1):
        t_v = t[k] * (1 + 0.609133 * q[k])

        ph_lev = p_half[k]
        ph_levplusone = p_half[k+1]

        if k == 0:
            dlog_p = np.log(ph_levplusone / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(ph_levplusone / ph_lev)
            alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        z_f = z_h + (t_v * R_d * alpha) # geopotential on half levels
        z_h = z_h + (t_v * R_d * dlog_p) # geopotential on full levels

        z_array[k] = z_f

    z_array = np.moveaxis(z_array, 0, levax)
    ds_era['z'] = (ds_era['t'].dims, z_array)

    return ds_era



def process_day(eradir_fx, eradir_rad, day, var_list, heights, g):
    """
    Read, postprocess, and regrid a single day of ERA5 data.

    Fuses what used to be a separate "read everything, then regrid everything"
    pass into one per-day step, so the full-resolution (137-level) raw arrays
    for a day are discarded as soon as that day has been regridded, instead of
    all days in the window being held in memory simultaneously.

    Parameters
    ----------
    eradir_fx : str
        Directory containing ERA files organized by date.
    eradir_rad : str
        Directory containing ERA files for radiation tendencies by date.
    day : datetime.datetime
        Day to process (00:00 of that day).
    var_list : list
        List of variable names to regrid.
    heights : numpy.ndarray
        Target height layers for regridding.
    g : float
        Gravitational acceleration, used to convert geopotential to height.

    Returns
    -------
    result : dict
        Mapping of variable name -> regridded array for this day.
    coords : dict
        longitude, latitude, and valid_time coordinates for this day.
    """
    
    hours = create_datelist(day, day + dt.timedelta(hours=23))

    rad_vars = [v for v in var_list if v in ('avg_ttswr', 'avg_ttlwr')]
    ds_rad_day = open_rad_day(eradir_rad, day, ['avg_ttswr', 'avg_ttlwr'])

    ds_day = xr.concat([postprocess_era(eradir_fx, ds_rad_day, t) for t in hours], dim='valid_time')

    ds_day['z'] = (ds_day['z'] - ds_day['z_sfc']) / g # geopotential -> height

    z_sel = ds_day['z'].values
    dp_sel = ds_day['dp'].values

    result = {}
    for var in var_list:
        var_sel = ds_day[var].values
        result[var] = vertint(z_sel, var_sel, dp_sel, heights, levax=1)

    coords = {
        'longitude': ds_day['longitude'].data,
        'latitude': ds_day['latitude'].data,
        'valid_time': ds_day['valid_time'].data,
    }

    del ds_day, z_sel, dp_sel

    return result, coords