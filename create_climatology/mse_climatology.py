#!/usr/bin/env python
"""
Hourly MSE climatology from ERA5 model levels (1991-2020), +/-10 day window.

Two stages:

  python mse_climatology.py compute  --workers 24
      One task per calendar day (MM-DD) over ALL years: reads each hourly file,
      computes MSE on your height layers, averages over the years and writes
      one small float32 .npy (24, height, lat, lon) per calendar day.
      Restartable: finished days are skipped.

  python mse_climatology.py finalize --workers 6
      One task per hour of day: applies the +/-WINDOW day running mean
      (weighted by the number of years actually available) across the
      per-day means and writes the final zarr store.

Calendar-day keyed (MM-DD), so "15 June" is always 5 June - 25 June, also in
leap years. Feb 29 is skipped (it only matters for the window buffer in
February). dayofyear in the output refers to a non-leap year (2001).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one core per worker

import argparse
import datetime as dt
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

# ------------------------------------------------------------------ settings
ERA_DIR = '/jetfs/shared-data/ECMWF/ERA5_glob_0.5deg_1h/'
WORK_DIR = '/jetfs/scratch/psplechtna/thesisdata/mse_climatology_2.0/work_2/'
OUT_ZARR = '/jetfs/scratch/psplechtna/thesisdata/mse_climatology_2.0/mse_hourly_climatology.zarr'

YEARS = range(1991, 2021)
SEASON_START = '05-16'
SEASON_END = '09-16'
WINDOW = 10
LAT_SLICE = slice(90, 20)       # ERA5 latitude is descending

CP, LV, G, R_D = 1005., 2.5e6, 9.80655, 287.06

HEIGHTS = np.array([0, 20, 50, 100, 200, 300, 400, 500, 
              600, 700, 800, 900, 1000, 1100, 1200, 1300,
              1400, 1500, 1600, 1700, 1800, 1900, 2000, 
              2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 
              3800, 4000, 4200, 4400, 4600, 4800, 5000, 5500, 
              6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,
              11000, 12000, 15000, 20000])
REF_YEAR = 2001                 # non-leap reference year for dayofyear


def calendar_days():
    """All MM-DD days needed: season +/- WINDOW (non-leap reference year)."""
    s = pd.Timestamp(f'{REF_YEAR}-{SEASON_START}') - pd.Timedelta(days=WINDOW)
    e = pd.Timestamp(f'{REF_YEAR}-{SEASON_END}') + pd.Timedelta(days=WINDOW)
    return pd.date_range(s, e)


# --------------------------------------------------------- vertical regridding
def vertint(z, var, dp, heights, levax=1):
    """Unchanged algorithm from the original code (layer means, proportional
    distribution of model levels across layer interfaces, dp-weighted)."""
    z_l = np.moveaxis(z, levax, 0)
    var_l = np.moveaxis(var, levax, 0)
    dp_l = np.moveaxis(dp, levax, 0)

    nlev = z_l.shape[0]
    nothers = int(np.prod(z_l.shape[1:])) if z_l.ndim > 1 else 1
    z_resh = z_l.reshape((nlev, nothers))
    var_resh = var_l.reshape((nlev, nothers))
    dp_resh = dp_l.reshape((nlev, nothers))

    ind = np.digitize(z_resh, heights) - 1
    values_grid = np.zeros((len(heights), nothers))
    weights_grid = np.zeros((len(heights), nothers))
    ind_other = np.arange(nothers)

    for i in range(nlev):
        ind_upper = np.minimum(len(heights) - 1, ind[i] + 1)
        ind_lower = np.maximum(0, ind[i] - 1)
        if i == nlev - 1:
            dist_zlower = z_resh[i] - 0.
            dist_hlower = z_resh[i] - 0.
        else:
            dist_zlower = z_resh[i] - z_resh[i + 1]
            dist_hlower = z_resh[i] - heights[ind[i]]
        if i == 0:
            dist_zupper = np.finfo(float).eps
            dist_hupper = np.finfo(float).eps
        else:
            dist_zupper = z_resh[i] - z_resh[i - 1]
            dist_hupper = z_resh[i] - heights[ind_upper]

        denom = dist_zlower / 2 - dist_zupper / 2
        fact_upper = np.maximum(-dist_zupper / 2 + dist_hupper, 0) / denom
        fact_lower = np.maximum(dist_zlower / 2 - dist_hlower, 0) / denom
        fact = 1 - fact_lower - fact_upper

        contrib = var_resh[i] * dp_resh[i]
        values_grid[ind[i], ind_other] += fact * contrib
        values_grid[ind_upper, ind_other] += fact_upper * contrib
        values_grid[ind_lower, ind_other] += fact_lower * contrib
        weights_grid[ind[i], ind_other] += fact * dp_resh[i]
        weights_grid[ind_upper, ind_other] += fact_upper * dp_resh[i]
        weights_grid[ind_lower, ind_other] += fact_lower * dp_resh[i]

    with np.errstate(invalid='ignore', divide='ignore'):
        out = values_grid[:-1] / weights_grid[:-1]   # top interface is out of domain
    return np.moveaxis(out.reshape((len(heights) - 1, *z_l.shape[1:])), 0, levax)


def geopotential_ml(t, q, p_half, z_sfc):
    """Geopotential on model levels (ECMWF compute_geopotential_on_ml), numpy only.
    t, q: (lev, lat, lon); p_half: (lev+1, lat, lon); z_sfc: (lat, lon)."""
    nlev = t.shape[0]
    z = np.empty_like(t)
    z_h = z_sfc.copy()
    for k in range(nlev - 1, -1, -1):
        t_v = t[k] * (1 + 0.609133 * q[k])
        ph, ph1 = p_half[k], p_half[k + 1]
        if k == 0:
            dlog_p = np.log(ph1 / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(ph1 / ph)
            alpha = 1. - (ph / (ph1 - ph)) * dlog_p
        z[k] = z_h + t_v * R_D * alpha
        z_h = z_h + t_v * R_D * dlog_p
    return z


# ------------------------------------------------------------------- reading
def _open(fname, keys, read_pv=False):
    bk = {'indexpath': ''}
    if read_pv:
        bk['read_keys'] = ['pv']
    return xr.open_dataset(fname, engine='cfgrib', backend_kwargs=bk, filter_by_keys=keys)


def read_era_hour(date):
    """Returns t, q, sp, z_sfc (float64, cropped to LAT_SLICE), pv, lat, lon."""
    f = ERA_DIR + date.strftime('%Y/%m/EA%Y%m%d%H')
    with _open(f, {'typeOfLevel': 'hybrid'}, read_pv=True) as ds:
        ds = ds.sel(latitude=LAT_SLICE)
        pv = ds['t'].attrs['GRIB_pv']
        t = ds['t'].values.astype(np.float64)
        q = ds['q'].values.astype(np.float64)
        lat, lon = ds['latitude'].values, ds['longitude'].values
    with _open(f, {'typeOfLevel': 'surface', 'dataType': 'an', 'edition': 1, 'shortName': 'z'}) as ds:
        z_sfc = ds['z'].sel(latitude=LAT_SLICE).values.astype(np.float64)
    with _open(f, {'typeOfLevel': 'surface', 'dataType': 'an', 'edition': 2}) as ds:
        sp = ds['sp'].sel(latitude=LAT_SLICE).values.astype(np.float64)
    return t, q, sp, z_sfc, pv, lat, lon


def mse_hour(date):
    """MSE on HEIGHTS[1:] for one hourly file -> (nheight, nlat, nlon)."""
    t, q, sp, z_sfc, pv, lat, lon = read_era_hour(date)
    n = len(pv) // 2
    hyai, hybi = np.asarray(pv[:n]), np.asarray(pv[n:2 * n])
    p_half = hyai[:, None, None] + hybi[:, None, None] * sp[None]
    dp = np.diff(p_half, axis=0)
    z = (geopotential_ml(t, q, p_half, z_sfc) - z_sfc) / G
    s = CP * t + LV * q                       # linear -> regrid once
    out = vertint(z, s, dp, HEIGHTS, levax=0)
    out += G * HEIGHTS[1:, None, None]
    return out, lat, lon


# --------------------------------------------------------------- stage 1
def _paths(day):
    tag = day.strftime('%m%d')
    return os.path.join(WORK_DIR, f'mean_{tag}.npy'), os.path.join(WORK_DIR, f'n_{tag}.npy')


def compute_day(day):
    """Mean over all YEARS of MSE for each hour of calendar day `day`."""
    f_mean, f_n = _paths(day)
    if os.path.exists(f_mean) and os.path.exists(f_n):
        return f'{day:%m-%d} skipped (done)'
    acc, cnt = None, np.zeros(24, dtype=np.int64)
    t0 = time.time()
    for iy, y in enumerate(YEARS, 1):
        for h in range(24):
            date = dt.datetime(y, day.month, day.day, h)
            try:
                field, lat, lon = mse_hour(date)
            except Exception as e:                      # missing / corrupt file
                print(f'  WARNING {date}: {type(e).__name__}: {e}', flush=True)
                continue
            if acc is None:
                acc = np.zeros((24, *field.shape))
                cf = os.path.join(WORK_DIR, 'coords.npz')
                if not os.path.exists(cf):
                    np.savez(cf + '.tmp.npz', lat=lat, lon=lon)
                    os.replace(cf + '.tmp.npz', cf)
            acc[h] += field
            cnt[h] += 1
        el = time.time() - t0
        print(f'  {day:%m-%d}: year {y} done ({iy}/{len(YEARS)}), '
              f'{el/60:.1f} min elapsed, ~{el/iy*len(YEARS)/60:.1f} min total, '
              f'{el/iy/24:.1f} s/file', flush=True)
    if acc is None:
        return f'{day:%m-%d} NO DATA'
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = (acc / cnt[:, None, None, None]).astype(np.float32)
    with open(f_n + '.tmp', 'wb') as fh:
        np.save(fh, cnt)
    os.replace(f_n + '.tmp', f_n)
    with open(f_mean + '.tmp', 'wb') as fh:
        np.save(fh, mean)
    os.replace(f_mean + '.tmp', f_mean)             # written last = "done" marker
    return f'{day:%m-%d} done (n/hour: {cnt.min()}-{cnt.max()})'


# --------------------------------------------------------------- stage 2
def window_mean(x, w, W):
    """Weighted running mean over 2W+1 entries of axis 0 (valid part only).
    x: (nd, ...) ; w: (nd,) sample counts. Returns (nd-2W, ...)."""
    nd = x.shape[0]
    wx = np.where(w.reshape((-1,) + (1,) * (x.ndim - 1)) > 0, x, 0.) * \
        w.reshape((-1,) + (1,) * (x.ndim - 1))
    cs = np.zeros((nd + 1,) + x.shape[1:])
    np.cumsum(wx, axis=0, out=cs[1:])
    cw = np.concatenate([[0], np.cumsum(w)]).astype(np.float64)
    num = cs[2 * W + 1:] - cs[:nd - 2 * W]
    den = (cw[2 * W + 1:] - cw[:nd - 2 * W]).reshape((-1,) + (1,) * (x.ndim - 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def _region_ds(arr):
    return xr.Dataset({'mse': (('dayofyear', 'hour', 'height', 'latitude', 'longitude'), arr)})

#---

def read_zsfc():
    """Static ERA5 surface geopotential (m2/s2) on the cropped grid."""
    f = ERA_DIR + dt.datetime(YEARS[0], 6, 15, 0).strftime('%Y/%m/EA%Y%m%d%H')
    with _open(f, {'typeOfLevel': 'surface', 'dataType': 'an',
                   'edition': 1, 'shortName': 'z'}) as ds:
        return ds['z'].sel(latitude=LAT_SLICE).values.astype(np.float64)


def smooth_hour(h):
    days = calendar_days()
    zsfc = read_zsfc()                                  # (lat, lon)
    counts = np.stack([np.load(_p(d, 'n'))[h] for d in days]).astype(np.float64)
    wv = np.clip(counts - 1, 0, None)
    mm = [np.load(_p(d, 'mean'), mmap_mode='r') for d in days]
    vm = [np.load(_p(d, 'var'),  mmap_mode='r') for d in days]
    nh = mm[0].shape[1]
    out = np.empty((len(days) - 2 * WINDOW, nh, *mm[0].shape[2:]), dtype=np.float32)
    outs = np.empty_like(out)
    for k in range(nh):
        xk = np.stack([a[h, k] for a in mm]).astype(np.float64)
        vk = np.stack([a[h, k] for a in vm]).astype(np.float64)
        out[:, k] = window_mean(xk, counts, WINDOW) + zsfc      # + g*z_sfc
        outs[:, k] = np.sqrt(window_mean(vk, wv, WINDOW))
    xr.Dataset({'mse': (DIMS, out[:, None]), 'mse_std': (DIMS, outs[:, None])}).to_zarr(
        OUT_ZARR, region={'dayofyear': slice(None), 'hour': slice(h, h + 1),
                          'height': slice(None), 'latitude': slice(None),
                          'longitude': slice(None)})
    return f'hour {h:02d} written'


#---



def smooth_hour(h):
    days = calendar_days()
    files = [_paths(d) for d in days]
    counts = np.stack([np.load(fn)[h] for _, fn in files]).astype(np.float64)
    x = np.stack([np.load(fm, mmap_mode='r')[h] for fm, _ in files])   # (nd, nh, ny, nx) f32
    out = np.empty((len(days) - 2 * WINDOW, *x.shape[1:]), dtype=np.float32)
    for k in range(x.shape[1]):                      # per height: keeps memory small
        out[:, k] = window_mean(x[:, k].astype(np.float64), counts, WINDOW)
    _region_ds(out[:, None]).to_zarr(
        OUT_ZARR, region={'dayofyear': slice(None), 'hour': slice(h, h + 1),
                          'height': slice(None), 'latitude': slice(None),
                          'longitude': slice(None)})
    return f'hour {h:02d} written'


def init_store():
    import dask.array as da
    days = calendar_days()[WINDOW:-WINDOW]
    c = np.load(os.path.join(WORK_DIR, 'coords.npz'))
    shape = (len(days), 24, len(HEIGHTS) - 1, len(c['lat']), len(c['lon']))
    ds = xr.Dataset(
        {'mse': (('dayofyear', 'hour', 'height', 'latitude', 'longitude'),
                 da.zeros(shape, chunks=(10, 1, shape[2], shape[3], shape[4]), dtype='float32'))},
        coords={'dayofyear': days.dayofyear.values, 'hour': np.arange(24),
                'height': HEIGHTS[1:], 'latitude': c['lat'], 'longitude': c['lon']},
        attrs={'description': f'Hourly MSE climatology {YEARS[0]}-{YEARS[-1]}, '
                              f'{SEASON_START} to {SEASON_END}, +/-{WINDOW} day window',
               'units': 'J/kg', 'dayofyear_reference_year': REF_YEAR})
    ds.to_zarr(OUT_ZARR, mode='w', compute=False)


# ------------------------------------------------------------------- driver
def _run(func, items, workers):
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(func, it) for it in items]
        for i, fu in enumerate(as_completed(futs), 1):
            print(f'[{i}/{len(futs)}] {fu.result()}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['compute', 'finalize'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--days', nargs='*', help='only these MM-DD (e.g. for SLURM arrays)')
    a = ap.parse_args()
    os.makedirs(WORK_DIR, exist_ok=True)

    if a.stage == 'compute':
        days = calendar_days()
        if a.days:
            days = [d for d in days if d.strftime('%m-%d') in a.days]
        _run(compute_day, list(days), a.workers)
    else:
        missing = [d for d in calendar_days() if not os.path.exists(_paths(d)[0])]
        if missing:
            raise SystemExit(f'{len(missing)} days not computed yet, e.g. {missing[0]:%m-%d}')
        init_store()
        _run(smooth_hour, list(range(24)), a.workers)