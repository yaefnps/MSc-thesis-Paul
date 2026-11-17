import eccodes
import numpy as np
import xarray as xr


# von Georg
g = 9.80655  # m/s^2
earth_radius = 6371000.0  # m
Rd = 287.06  # J/(kg K)

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

def average_to_flexpart_layers(source, boundaries, upper_bounds):
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
        layer_means.append((source * overlap).sum(axis=0) / (top - bottom))
        bottom = top

    return np.stack(layer_means)