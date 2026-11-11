import xarray as xr
import numpy as np
import dask

path = '/jetfs/scratch/psplechtna/thesisdata/sfcfluxes/'
years = np.arange(1991, 2021, 1)

all_fluxes = xr.open_mfdataset([path+f'fluxes_per_sec_{year}.nc' for year in years]).sel(latitude=slice(90,0)).drop_vars(['number', 'expver'])
all_fluxes = all_fluxes.sel(valid_time=all_fluxes.valid_time.dt.month.isin([5, 6, 7, 8]))
all_fluxes = all_fluxes.chunk({"valid_time": -1, "latitude": 20, "longitude": 120})

t = all_fluxes.valid_time.dt
all_fluxes = all_fluxes.assign_coords(mdh=("valid_time", (t.month * 10000 + t.day * 100 + t.hour).data))

clim = all_fluxes[["slhf", "sshf"]].groupby("mdh").mean("valid_time")
clim.to_zarr(path + 'clim_fluxes.zarr')