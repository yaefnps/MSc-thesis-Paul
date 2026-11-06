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