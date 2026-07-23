from mehori_climate import LatHgtPlot, FigurePanel
import matplotlib.pyplot as plt
import xarray as xr

# open data
ds = xr.open_dataset('../../data/u.djf.nc',decode_times=False)

# 0. read data
u = ds["ua"]
my_data = (
    u[0,:,:,:]
    .sel(lat=slice(90,0))
    .sel(plev=slice(1000, 100))
    .sel(lon=120, method="nearest")
)

# print(my_data.dims)
# print(my_data.shape)
# print(ds["lat"])
# exit

# 1. Create the figure panel
fig   = FigurePanel(figsize=(6, 4))
plot1 = LatHgtPlot(fig)

# 2. Chain your methods
(plot1
    .add_shade(my_data, vmin=-50, vmax=50, interval=5, color="RdBu_r")
    .add_gridlines()
#   .add_colorbar() 
    .add_title(title="Difference Map")
    .add_corner(left_text="Scenario A", right_text="[DJF]")
)

# 3. Render plot
fig.render(ofile="output.test05.png")


