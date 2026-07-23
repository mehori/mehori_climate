from mehori_climate import LatlonPlot, FigurePanel
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# open data
ds1 = xr.open_dataset('../../data/slp.djf.nc', decode_times=False)
ds2 = xr.open_dataset('../../data/u.djf.nc', decode_times=False)
ds3 = xr.open_dataset('../../data/v.djf.nc', decode_times=False)

# 0. read data
slp = ds1["psl"]
u   = ds2["ua"]
v   = ds3["va"]
my_data = slp[0,:,:]
uval = u[0,3,:,:]
vval = v[0,3,:,:]

speed = np.sqrt(uval**2 + vval**2)
threshold = 5.0  # e.g. m/s
u_masked = uval.where(speed >= threshold)
v_masked = vval.where(speed >= threshold)

# 1. Create the figure panel
fig   = FigurePanel(figsize=(16, 6))
plot1 = LatlonPlot(fig)

# 2. Chain your methods
(plot1
    .add_shade(my_data, vnum=17, color="RdBu_r", withLine=True, linecolor="white")
    .add_vector(u_masked,v_masked)
    .add_coastlines(fillland=False)
    .add_gridlines()
    .add_colorbar() 
    .add_title(title="Difference Map")
    .add_corner(left_text="Scenario A", right_text="[DJF]")
)

# 3. Render plot
fig.render(ofile="output.test03.png")

