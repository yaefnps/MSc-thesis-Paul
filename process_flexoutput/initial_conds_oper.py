from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
import eccodes

sys.path.insert(0, "/jetfs/home/psplechtna/thesis/preprocess_flexout/")
from read_binary import read_namelists, read_gridini

run = sys.argv[1]

# ---------------------------------------------------------------- constants
cp = 1005.0
Lv = 2.5e6
g = 9.80655
Rd = 287.06
earth_radius = 6371000.0

# ------------------------------------------------------------------- paths
run_dir = Path("/jetfs/scratch/psplechtna/flexpart_output") / run
out_dir = Path("/jetfs/scratch/psplechtna/flexpart_output") / run / "processed_data"
path_clim = "/jetfs/scratch/psplechtna/thesisdata/mse_climatology_4.0/"
filename_clim = "mse_hourly_climatology.zarr"
met_dir = Path("/jetfs/shared-data/ECMWF/ERA5_glob_0.5deg_1h/")

# ----------------------------------------------------------------- helpers
def wrap_lon(lon):
    """Longitudes to [-180, 180)."""
    return ((lon + 180) % 360) - 180


def as_array(da, dims):
    return da.transpose(*dims).values.astype("float64")


def geometric_altitude(geopotential):
    geopotential_height = geopotential / g
    return (earth_radius * geopotential_height) / (earth_radius - geopotential_height)


def read_hybrid_coefficients(filename):
    with open(filename, "rb") as file:
        while True:
            message = eccodes.codes_grib_new_from_file(file)
            if message is None:
                raise ValueError("Kein Temperaturfeld gefunden.")
            try:
                if eccodes.codes_get(message, "paramId") == 130:
                    coefficients = np.asarray(
                        eccodes.codes_get_array(message, "pv"), dtype="float"
                    )
                    break
            finally:
                eccodes.codes_release(message)

    assert coefficients.size == 276
    return coefficients[:138], coefficients[138:]  # a_half, b_half


def read_met(met_file, latitude, longitude):
    """Read t, q, sp, z from the ERA5 GRIB file on the given grid."""
    fields = []
    for param_id in (130, 133, 134, 129):
        with xr.open_dataset(
            met_file,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"paramId": param_id}, "indexpath": ""},
        ) as ds:
            ds = ds.assign_coords(longitude=wrap_lon(ds.longitude)).sortby("longitude")
            ds = ds.sel(latitude=latitude, longitude=longitude)
            fields.append(ds.drop_vars("surface", errors="ignore").load())
    return xr.merge(fields, join="exact").sortby("hybrid")


def calculate_geometry(met, a_half, b_half):
    """Full/half-level heights above ground and ground altitude (met is sorted by hybrid)."""
    temperature = as_array(met.t, ("hybrid", "latitude", "longitude"))
    humidity = as_array(met.q, ("hybrid", "latitude", "longitude"))
    surface_pressure = as_array(met.sp, ("latitude", "longitude"))
    surface_geopotential = as_array(met.z, ("latitude", "longitude"))

    virtual_temperature = temperature * (1.0 + 0.609133 * humidity)
    pressure_half = a_half[:, None, None] + b_half[:, None, None] * surface_pressure

    n_levels = temperature.shape[0]  # 137
    geopotential_full = np.empty_like(temperature)
    geopotential_half = np.empty_like(pressure_half)
    geopotential_half[n_levels] = 0.0

    for level in range(n_levels, 0, -1):
        p_lower = pressure_half[level]
        p_upper = pressure_half[level - 1]
        rt = Rd * virtual_temperature[level - 1]

        if level == 1:
            dlog_p = np.log(p_lower / 0.1)
            alpha = np.log(2.0)
        else:
            dlog_p = np.log(p_lower / p_upper)
            alpha = 1.0 - (p_upper / (p_lower - p_upper) * dlog_p)

        geopotential_full[level - 1] = geopotential_half[level] + rt * alpha
        geopotential_half[level - 1] = geopotential_half[level] + rt * dlog_p

    geopotential_half[0] = np.inf

    ground_altitude = geometric_altitude(surface_geopotential)
    height_full = geometric_altitude(surface_geopotential + geopotential_full) - ground_altitude

    height_half = np.empty_like(geopotential_half)
    height_half[1:] = geometric_altitude(surface_geopotential + geopotential_half[1:]) - ground_altitude
    height_half[0] = np.inf

    return height_full, height_half, ground_altitude


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


# ------------------------------------------------------- read grid_initial
times_ini, times_fin, heights, lat, lon, age = read_namelists(run_dir)
ds_ini = read_gridini(run_dir, times_ini, heights, lat, lon)

# duplicate the westernmost column at -179.5 to close the wrap-around
wrap1 = ds_ini.sel(longitude=-179).assign_coords(longitude=-179.5)
ds_ini = xr.concat([wrap1, ds_ini], dim="longitude")
ds_ini = ds_ini.assign_coords(longitude=wrap_lon(ds_ini.longitude)).sortby("longitude")

# --------------------------------------------- MSE on the FLEXPART grid
component_names = ["cpT", "Lvq", "pot"]
all_components = component_names + ["MSE"]

n_releases = len(times_ini)

grid_raw = np.full((n_releases, len(all_components)), np.nan)
sensitivity_sums = np.full(n_releases, np.nan)
initial_times = [pd.Timestamp(t) for t in times_ini]
final_times = [pd.Timestamp(t) for t in times_fin]
assert len(initial_times) == len(final_times), "times_ini and times_fin differ in length"

HYB = ("hybrid", "latitude", "longitude")

for index, initial_time in enumerate(initial_times):
    sensitivity = ds_ini.sensitivity.isel(valid_time=index, drop=True)
    sensitivity_sums[index] = sensitivity.sum().item()

    met_file = (
        met_dir
        / initial_time.strftime("%Y/%m")
        / initial_time.strftime("EA%Y%m%d%H")
    )

    met = read_met(met_file, sensitivity.latitude, sensitivity.longitude)
    a_half, b_half = read_hybrid_coefficients(met_file)
    height_full, height_half, ground_altitude = calculate_geometry(met, a_half, b_half)
    dp = pressure_thickness(met, a_half, b_half)

    energy_fields = {
        "cpT": cp * as_array(met.t, HYB),
        "Lvq": Lv * as_array(met.q, HYB),
        "pot": g * (height_full + ground_altitude),
    }

    # map to FLEXPART layers (mass-weighted) and weight with the sensitivity
    for k, name in enumerate(component_names):
        layer_energy = xr.DataArray(
            average_to_layers_massweighted(
                energy_fields[name], height_half, sensitivity.height.values, dp
            ),
            dims=["height", "latitude", "longitude"],
            coords={
                "height": sensitivity.height,
                "latitude": sensitivity.latitude,
                "longitude": sensitivity.longitude,
            },
        )
        grid_raw[index, k] = (layer_energy * sensitivity).sum(skipna=False).item()

    grid_raw[index, -1] = grid_raw[index, :-1].sum()  # MSE

    print(f"Release {index + 1}/{n_releases} fertig", flush=True)

init_conds = xr.Dataset(
    data_vars={
        "grid_raw_Jkg": (("release", "component"), grid_raw),
        "sensitivity_sum": ("release", sensitivity_sums),
    },
    coords={
        "release": np.arange(1, n_releases + 1),
        "component": all_components,
        "initial_time": ("release", initial_times),
        "final_time": ("release", final_times),
    },
)
init_conds["grid_normalized_Jkg"] = init_conds.grid_raw_Jkg / init_conds.sensitivity_sum
init_conds.grid_raw_Jkg.attrs["units"] = "J kg-1"
init_conds.grid_normalized_Jkg.attrs["units"] = "J kg-1"

# ------------------------------------------------ climatological MSE
clim = xr.open_mfdataset(
    path_clim + filename_clim, engine="zarr", concat_dim="time", combine="nested"
)
clim_mse = clim["mse"].sel(time=0, drop=True)

# pointwise (doy, hour) selection along "release"
i_doy = clim_mse.indexes["dayofyear"].get_indexer(init_conds.initial_time.dt.dayofyear.values)
i_hour = clim_mse.indexes["hour"].get_indexer(init_conds.initial_time.dt.hour.values)
assert (i_doy >= 0).all() and (i_hour >= 0).all(), "doy/hour not found in climatology"

clim_mse = clim_mse.isel(
    dayofyear=xr.DataArray(i_doy, dims="release"),
    hour=xr.DataArray(i_hour, dims="release"),
).load()

clim_mse = clim_mse.assign_coords(
    release=init_conds.release.values,
    initial_time=("release", init_conds.initial_time.values),
    longitude=wrap_lon(clim_mse.longitude),
).sortby("longitude")

# ------------------------------------- weight climatology with sensitivity
sens = (
    ds_ini.sensitivity
    .sel(valid_time=init_conds.initial_time.values)
    .rename(valid_time="release")
    .assign_coords(release=init_conds.release.values)
)

lat_common = sens.indexes["latitude"].intersection(clim_mse.indexes["latitude"])
lon_common = sens.indexes["longitude"].intersection(clim_mse.indexes["longitude"])
sens = sens.sel(latitude=lat_common, longitude=lon_common)
clim_mse = clim_mse.sel(latitude=lat_common, longitude=lon_common)

# fail loudly if the grids differ (height only exists on sens)
xr.align(sens, clim_mse, join="exact", exclude=["height"])

# the intersection must not cut into the footprint, otherwise the
# normalisation by sensitivity_sum is not comparable to the grid result
kept = sens.sum(["height", "latitude", "longitude"])
#np.testing.assert_allclose(kept.values, init_conds.sensitivity_sum.values, rtol=1e-5)

clim_raw = (sens * clim_mse).sum(dim=["height", "latitude", "longitude"], skipna=False)

init_conds["clim"] = clim_raw / init_conds.sensitivity_sum
init_conds["init_anom"] = (
    init_conds.grid_normalized_Jkg.sel(component="MSE") - init_conds.clim
)
init_conds.clim.attrs["units"] = "J kg-1"
init_conds.init_anom.attrs["units"] = "J kg-1"

out_dir.mkdir(parents=True, exist_ok=True)
init_conds.to_netcdf(out_dir / "initial_conditions.nc")