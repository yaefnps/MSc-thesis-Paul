import numpy as np
import dask
import glob
import psutil, os
import xarray as xr
import sys
from dask.diagnostics import ProgressBar
import time

print('running', __file__, flush=True)

def mem(tag):
    print(f'[{tag}] RSS = {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB', flush=True)
    
import threading, time

def _monitor(interval=60):
    while True:
        mem('monitor')
        time.sleep(interval)

threading.Thread(target=_monitor, daemon=True).start()

def centered_diff(da, dim='time'):
    """
    Computes a centered (second-order) finite difference along `dim`:
    
        f'(t) ~ (f(t+1) - f(t-1)) / 2
    
    Unlike xarray's .diff(), this keeps the same length/coordinates as the
    input array, with NaN at the first and last step along `dim` (since a
    centered difference needs a neighbor on both sides).
    
    Inputs:
    -----------
        da: [xarray.DataArray] the array to differentiate
        dim: [str] the dimension to differentiate along (default 'time')
    
    Returns:
    -----------
        [xarray.DataArray] same shape as da
    """
    return (da.shift({dim: -1}) - da.shift({dim: 1})) / 2

run = sys.argv[1]
height_str = sys.argv[2]
height = float(height_str)

path = '/jetfs/scratch/psplechtna/flexpart_output/oper/' + run + '/'
filename = glob.glob(os.path.join(path, 'grid_time_*.nc'))[0]
path_fluxes = '/jetfs/scratch/psplechtna/thesisdata/sfcfluxes/'
filename_fluxes = f'fluxes_per_sec_full2024.nc'

outdir      = path + 'processed_data/'


dask.config.set(scheduler='threads',
                num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))

# --- FLEXPART: explicit chunks, only what is needed ---------------------
# height stays whole (it is reduced), chunk over time and pointspec.
# Tune pointspec so one chunk is ~100-200 MB.
mem('start')
data = xr.open_dataset(
    filename,
    chunks={'time': 1, 'pointspec': 1, 'height': -1,
            'latitude': -1, 'longitude': -1},
)
# add missing longitude data - linear interpolation between -179 and 180
west = data.sel(longitude=-179).load()
east = data.sel(longitude=180).load()

wrapper_value = 0.5 * (west.drop_vars("longitude") + east.drop_vars("longitude"))
wrapper_value = wrapper_value.expand_dims(longitude=[-179.5])

data = xr.concat(
    [wrapper_value, data],
    dim="longitude",
    data_vars="minimal",
    coords="minimal",
    compat="override",
)
data = data.assign_coords(longitude=data.longitude % 360).sortby("longitude").sortby("latitude")
data = data.sel(latitude=slice(30, 90))      # subset BEFORE any heavy work
data = data.chunk({'time': 1, 'pointspec': 1, 'height': -1, 'latitude': -1, 'longitude': -1})
data_mr = data['spec001_mr'].isel(nageclass=0)

# --- ERA5 fields: merge, subset, load, interpolate once -----------------
def load_era5(path, names):
    ds = xr.open_dataset(path)[names].rename({'valid_time': 'time'})
    ds = ds.sel(time=slice(data.time.min().values, data.time.max().values))  # contiguous block first
    ds = ds.sel(latitude=data.latitude.values, longitude=data.longitude.values)
    ds = ds.sel(time=data.time.values)
    return ds.load()

t = time.time()
fl  = load_era5(path_fluxes + filename_fluxes, ['slhf', 'sshf']); print('fluxes loaded', time.time()-t, flush=True)
fl = fl.chunk({'time': 1})

mem('after era5')

# choose all heights below height
released = spec.isel(height=(spec.height <= height).values).sum('height')
# --- # sfc fluxes
rel_lh = (-fl['slhf'] * released / height).squeeze().rename('lh_bl')
rel_sh = (-fl['sshf'] * released / height).squeeze().rename('sh_bl')

out_chunks = {'time': 1, 'pointspec': -1, 'latitude': -1, 'longitude': -1}

outputs = {
    f'lh_{height}':     rel_lh,
    f'sh_{height}':     rel_sh,
}

def rechunk(da):
    return da.chunk({k: v for k, v in out_chunks.items() if k in da.dims})

def clean(ds):
    # drop leftover netCDF encoding that can conflict with zarr
    for v in ds.variables.values():
        v.encoding = {}
    return ds

for name, da in outputs.items():
    clean(rechunk(da).to_dataset(name=da.name)).to_zarr(
        outdir + name + '.zarr', mode='w', compute=False)

def write_batch(da, name, sl):
    ds = clean(rechunk(da.isel(time=sl)).to_dataset(name=da.name))
    # region writes need every variable to have a time dimension
    ds = ds.drop_vars([v for v in ds.variables if 'time' not in ds[v].dims])
    return ds.to_zarr(outdir + name + '.zarr', region={'time': sl}, compute=False)

nt, step = data.sizes['time'], 12
t0 = time.time()
for i in range(0, nt, step):
    sl = slice(i, min(i + step, nt))
    dask.compute(*[write_batch(da, n, sl) for n, da in outputs.items()])
    mem(f'after batch {sl.start}')
    print(f'done {sl.start}-{sl.stop} of {nt}  ({(time.time()-t0)/60:.1f} min)', flush=True)
mem('after compute')