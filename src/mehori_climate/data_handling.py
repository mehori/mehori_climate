import xarray as xr
from   scipy import stats

def calc_tdiff(v1, v2):
    # 1. Calculate the mean difference
    diff = v1.mean('time') - v2.mean('time')
    
    # 2. Perform the two-tailed t-test (omit NaNs)
    t_stat, p_val = stats.ttest_ind(v1, v2, axis=0, nan_policy='omit')
    
    # 3. Create a DataArray for p-values to retain lat/lon coordinates
    ttest = xr.DataArray(p_val, coords=diff.coords)
    return diff, ttest



