#!/usr/bin/env python
"""
Hourly MSE climatology (mean + std) from ERA5 model levels (1991-2020), +/-10 day window.

Vertical regridding is identical to regrid_rad_g.py / the initial-condition script:
  - heights and half-level geometry from era5_to_flex_helpers (read_met, calculate_geometry)
  - mass-weighted (dp) layer means with exact overlap handling
  - MSE = cp*T + Lv*q + g*(height_full + ground_altitude), regridded in ONE call
    (the layer mean is linear, so this equals summing the regridded components)
  - the potential-energy term is therefore the layer MEAN of g*z (as in the initial
    conditions), already including the ground altitude. Nothing is added in 'finalize'.

Two stages:

  python create_mse_clim.py compute  --workers 24
      One task per calendar day (MM-DD) over ALL years: reads each hourly file,
      computes MSE on the height layers and, for each hour of day, the
      multi-year mean, variance (ddof=1) and sample count. Writes small float32
      .npy files (24, height, lat, lon) per calendar day.
      Restartable: finished days are skipped.

  python create_mse_clim.py finalize --workers 6
      One task per hour of day: applies the +/-WINDOW day window (weighted by the
      number of years actually available) to the per-day means, and combines the
      per-day variances into the variance of ALL values in the window
      (30 years x 21 days = 630 values for a full window).

Calendar-day keyed (MM-DD), so "15 June" is always 5 June - 25 June, also in
leap years. Feb 29 is skipped. dayofyear in the output refers to a non-leap year (2001).

mse_std (STD_KIND):
  'total'  : std of all values in the window about the window mean (ddof=1).
             Includes interannual + synoptic variability AND the seasonal drift
             across the window.
  'within' : pooled std about each calendar day's own multi-year mean (old behaviour);
             no seasonal drift.
Neither contains the diurnal cycle (every hour of day is treated separately).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one core per worker

import argparse
import datetime as dt
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from era5_to_flex_helpers import (
    read_hybrid_coefficients,
    read_met,
    calculate_geometry,
    wrap_lon,
    as_array,
)

# ------------------------------------------------------------------ settings
ERA_DIR = '/jetfs/shared-data/ECMWF/ERA5_glob_0.5deg_1h/'
WORK_DIR = '/jetfs/scratch/psplechtna/thesisdata/mse_climatology_4.0/work/'
OUT_ZARR = '/jetfs/scratch/psplechtna/thesisdata/mse_climatology_4.0/mse_hourly_climatology.zarr'

YEARS = range(1991, 2021)
SEASON_START = '05-16'
SEASON_END = '09-16'
WINDOW = 10
LAT_SLICE = slice(90, 20)       # ERA5 latitude is descending
STD_KIND = 'total'              # 'total' (all window values) or 'within' (old pooled version)

CP, LV = 1005., 2.5e6
G = 9.80655

HEIGHTS = np.array([0, 20, 50, 100, 200, 300, 400, 500,
                    600, 700, 800, 900, 1000, 1100, 1200, 1300,
                    1400, 1500, 1600, 1700, 1800, 1900, 2000,
                    2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600,
                    3800, 4000, 4200, 4400, 4600, 4800, 5000, 5500,
                    6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,
                    11000, 12000, 15000, 20000], dtype=float)
REF_YEAR = 2001                 # non-leap reference year for dayofyear
DIMS = ('dayofyear', 'hour', 'height', 'latitude', 'longitude')
HYB = ('hybrid', 'latitude', 'longitude')
REF_FILE = ERA_DIR + dt.datetime(YEARS[0], 6, 15, 0).strftime('%Y/%m/EA%Y%m%d%H')


def calendar_days():
    """All MM-DD days needed: season +/- WINDOW (non-leap reference year)."""
    s = pd.Timestamp(f'{REF_YEAR}-{SEASON_START}') - pd.Timedelta(days=WINDOW)
    e = pd.Timestamp(f'{REF_YEAR}-{SEASON_END}') + pd.Timedelta(days=WINDOW)
    return pd.date_range(s, e)


# ----------------------------------------------- grid + coefficients
@lru_cache(maxsize=None)
def _grid():
    """Cropped lat/lon (lon wrapped to [-180, 180) and sorted, as in read_met)."""
    with xr.open_dataset(
        REF_FILE, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'paramId': 130}, 'indexpath': ''},
    ) as ds:
        ds = ds.assign_coords(longitude=wrap_lon(ds.longitude)).sortby('longitude')
        ds = ds.sel(latitude=LAT_SLICE)
        return ds.latitude.values.copy(), ds.longitude.values.copy()


@lru_cache(maxsize=None)
def _hybrid_coeffs():
    return read_hybrid_coefficients(REF_FILE)


# --------------------------------------------------------- vertical regridding
def pressure_thickness(met, a_half, b_half):
    """Pressure thickness dp (Pa) of each model level, index 0 = model top."""
    surface_pressure = as_array(met.sp, ("latitude", "longitude"))
    pressure_half = a_half[:, None, None] + b_half[:, None, None] * surface_pressure
    return np.diff(pressure_half, axis=0)


def average_to_layers_massweighted(source, boundaries, upper_bounds, dp):
    """
    Mass-weighted layer means with exact overlap handling.
    Each model level contributes with the part of its air mass (dp) that lies
    inside the layer (uniform density within a level assumed).
    source (nlev, ...), boundaries (nlev+1, ...) half-level heights, boundaries[0] = inf.
    """
    thickness = boundaries[:-1] - boundaries[1:]  # inf for level 0 -> fraction 0
    layer_means = []
    bottom = 0.0

    for top in upper_bounds:
        overlap = np.maximum(
            0.0,
            np.minimum(boundaries[:-1], top) - np.maximum(boundaries[1:], bottom),
        )
        np.testing.assert_allclose(
            overlap.sum(axis=0), top - bottom, rtol=1e-8, atol=1e-6
        )
        w = dp * overlap / thickness
        layer_means.append((source * w).sum(axis=0) / w.sum(axis=0))
        bottom = top

    return np.stack(layer_means)


def mse_hour(date):
    """Mass-weighted MSE on HEIGHTS[1:] for one hourly file -> (nheight, nlat, nlon)."""
    lat, lon = _grid()
    a_half, b_half = _hybrid_coeffs()
    met = read_met(ERA_DIR + date.strftime('%Y/%m/EA%Y%m%d%H'), lat, lon)
    height_full, height_half, ground_altitude = calculate_geometry(met, a_half, b_half)
    dp = pressure_thickness(met, a_half, b_half)

    mse = (CP * as_array(met.t, HYB)
           + LV * as_array(met.q, HYB)
           + G * (height_full + ground_altitude))
    out = average_to_layers_massweighted(mse, height_half, HEIGHTS[1:], dp)
    return out, lat, lon


# --------------------------------------------------------------- stage 1
def _p(day, kind):
    return os.path.join(WORK_DIR, f'{kind}_{day:%m%d}.npy')


def compute_day(day):
    """Per calendar day: mean, variance (ddof=1) and counts for every hour,
    over all YEARS."""
    f_mean, f_var, f_n = (_p(day, k) for k in ('mean', 'var', 'n'))
    if all(os.path.exists(f) for f in (f_mean, f_var, f_n)):
        return f'{day:%m-%d} skipped (done)'

    ny = len(YEARS)
    cnt = np.zeros(24, dtype=np.int64)
    mean = var = None
    t0 = time.time()

    for h in range(24):
        stack = None
        for iy, y in enumerate(YEARS):
            date = dt.datetime(y, day.month, day.day, h)
            try:
                field, lat, lon = mse_hour(date)
            except Exception as e:                      # missing / corrupt file
                print(f'  WARNING {date}: {type(e).__name__}: {e}', flush=True)
                continue
            if stack is None:
                stack = np.full((ny, *field.shape), np.nan, dtype=np.float32)
                cf = os.path.join(WORK_DIR, 'coords.npz')
                if not os.path.exists(cf):
                    np.savez(cf + '.tmp.npz', lat=lat, lon=lon)
                    os.replace(cf + '.tmp.npz', cf)
            stack[iy] = field
            cnt[h] += 1

        if stack is None:
            continue
        if mean is None:                                # first hour with data
            mean = np.full((24, *stack.shape[1:]), np.nan, dtype=np.float32)
            var = np.full_like(mean, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mean[h] = np.nanmean(stack, axis=0, dtype=np.float64)
            if cnt[h] > 1:
                var[h] = np.nanvar(stack, axis=0, ddof=1, dtype=np.float64)
        print(f'  {day:%m-%d}: hour {h:02d} done, {(time.time()-t0)/60:.1f} min', flush=True)

    if mean is None:
        return f'{day:%m-%d} NO DATA'

    for fn, arr in ((f_n, cnt), (f_var, var), (f_mean, mean)):   # mean last = done marker
        with open(fn + '.tmp', 'wb') as fh:
            np.save(fh, arr)
        os.replace(fn + '.tmp', fn)
    return f'{day:%m-%d} done (n/hour: {cnt.min()}-{cnt.max()})'


# --------------------------------------------------------------- stage 2
def window_mean(x, w, W):
    """Weighted running mean over 2W+1 entries of axis 0 (valid part only).
    x: (nd, ...) ; w: (nd,) sample counts. Returns (nd-2W, ...).
    NaN entries get zero weight instead of poisoning the cumulative sums."""
    nd = x.shape[0]
    wb = np.broadcast_to(w.reshape((-1,) + (1,) * (x.ndim - 1)), x.shape)
    valid = np.isfinite(x) & (wb > 0)
    wx = np.where(valid, x, 0.) * wb
    ww = np.where(valid, wb, 0.)
    cs = np.zeros((nd + 1,) + x.shape[1:])
    cw = np.zeros((nd + 1,) + x.shape[1:])
    np.cumsum(wx, axis=0, out=cs[1:])
    np.cumsum(ww, axis=0, out=cw[1:])
    num = cs[2 * W + 1:] - cs[:nd - 2 * W]
    den = cw[2 * W + 1:] - cw[:nd - 2 * W]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def window_sum(w, W):
    """Running sum over 2W+1 entries of a 1D array (valid part only)."""
    cs = np.concatenate(([0.], np.cumsum(w)))
    return cs[2 * W + 1:] - cs[:len(w) - 2 * W]


def window_stats(x, v, counts, W, kind=STD_KIND):
    """
    Window mean and std from per-day means x, per-day variances v (ddof=1) and
    per-day sample counts. x, v: (nd, ...); counts: (nd,).

    'total' : variance of ALL values in the window about the window mean M,
              var = [ sum (n_d-1) v_d + sum n_d (m_d - M)^2 ] / (N - 1)
    'within': pooled variance about each day's own mean, sum (n_d-1) v_d / sum (n_d-1)
    """
    wv = np.clip(counts - 1, 0, None)
    mean = window_mean(x, counts, W)
    within = window_mean(v, wv, W)
    if kind == 'within':
        return mean, np.sqrt(np.maximum(within, 0))

    shape = (-1,) + (1,) * (x.ndim - 1)
    n_win = window_sum(counts, W).reshape(shape)
    k_win = window_sum(wv, W).reshape(shape)
    # centre first: E[x^2] - M^2 on values of ~3e5 J/kg would lose precision
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.nanmean(x, axis=0, keepdims=True)
    xc = x - ref
    between = n_win * (window_mean(xc ** 2, counts, W) - window_mean(xc, counts, W) ** 2)
    var = (k_win * within + between) / (n_win - 1)
    return mean, np.sqrt(np.maximum(var, 0))


def smooth_hour(h):
    days = calendar_days()
    counts = np.array([np.load(_p(d, 'n'))[h] for d in days], dtype=np.float64)
    x = np.stack([np.load(_p(d, 'mean'), mmap_mode='r')[h] for d in days])
    v = np.stack([np.load(_p(d, 'var'), mmap_mode='r')[h] for d in days])
    out = np.empty((len(days) - 2 * WINDOW, *x.shape[1:]), dtype=np.float32)
    outs = np.empty_like(out)
    for k in range(x.shape[1]):                         # per height: keeps memory small
        m, s = window_stats(x[:, k].astype(np.float64), v[:, k].astype(np.float64),
                            counts, WINDOW)
        out[:, k], outs[:, k] = m, s
    xr.Dataset({'mse': (DIMS, out[:, None]), 'mse_std': (DIMS, outs[:, None])}).to_zarr(
        OUT_ZARR, region={'dayofyear': slice(None), 'hour': slice(h, h + 1),
                          'height': slice(None), 'latitude': slice(None),
                          'longitude': slice(None)})
    return f'hour {h:02d} written'


def init_store():
    import dask.array as da
    days = calendar_days()[WINDOW:-WINDOW]
    c = np.load(os.path.join(WORK_DIR, 'coords.npz'))
    shape = (len(days), 24, len(HEIGHTS) - 1, len(c['lat']), len(c['lon']))
    mk = lambda: (DIMS, da.zeros(shape, chunks=(10, 1, shape[2], shape[3], shape[4]),
                                 dtype='float32'))
    std_text = ('std of all values in the window (years x window days) about the window mean, '
                'incl. seasonal drift across the window' if STD_KIND == 'total' else
                'pooled over years and window days about each calendar day\'s own multi-year '
                'mean (no seasonal drift)')
    ds = xr.Dataset(
        {'mse': mk(), 'mse_std': mk()},
        coords={'dayofyear': days.dayofyear.values, 'hour': np.arange(24),
                'height': HEIGHTS[1:], 'latitude': c['lat'], 'longitude': c['lon']},
        attrs={'description': f'Hourly MSE climatology {YEARS[0]}-{YEARS[-1]}, '
                              f'{SEASON_START} to {SEASON_END}, +/-{WINDOW} day window. '
                              'Mass-weighted (dp) layer means with exact overlap handling; '
                              'potential energy = layer mean of g*(height above ground + ground altitude). '
                              f'mse_std: {std_text}; the diurnal cycle is not included.',
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
        missing = [d for d in calendar_days()
                   if not all(os.path.exists(_p(d, k)) for k in ('mean', 'var', 'n'))]
        if missing:
            raise SystemExit(f'{len(missing)} days not computed yet, e.g. {missing[0]:%m-%d}')
        init_store()
        _run(smooth_hour, list(range(24)), a.workers)