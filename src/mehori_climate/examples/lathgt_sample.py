"""
Sample script: latitude-height (pressure) cross-section plot using
LatHgtPlot.

Demonstrates a typical zonal-mean cross section (e.g. wind or temperature
anomaly vs. latitude and pressure), built the same way as the lat-lon and
polar examples: chain the panel's own methods, then call render() once via
FigurePanel.

Replace make_sample_data() with your own xr.open_dataset(...) call --
LatHgtPlot.add_shade() just needs a 2D xarray DataArray with a 'lat' dim
and a vertical pressure dim (auto-detected from level/lev/plev/pressure/pres).
"""
import numpy as np
import xarray as xr

from mehori_climate import LatHgtPlot, FigurePanel


def make_sample_data():
    """ Synthetic zonal-mean zonal wind anomaly (lev x lat), standing in
    for real data (e.g. `ds["ua"].mean("lon")` from a reanalysis file). """
    lat = np.linspace(-90, 90, 73)
    lev = np.array([1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10])

    LAT, LEV = np.meshgrid(lat, lev)
    # A rough double-jet-like pattern: anomaly centered near 45 deg in each
    # hemisphere (opposite sign), strongest in the upper troposphere.
    data = (
        15
        * np.exp(-((np.abs(LAT) - 45) ** 2) / (2 * 20 ** 2))
        * np.exp(-((np.log(LEV / 1000.0) + 1.2) ** 2) / (2 * 0.6 ** 2))
        * np.sign(LAT)
    )

    return xr.DataArray(
        data, coords={"lev": lev, "lat": lat}, dims=["lev", "lat"],
        name="u_anom", attrs={"units": "m/s"},
    )


def main():
    data = make_sample_data()
    # To use real data instead:
    # ds   = xr.open_dataset("path/to/file.nc")
    # data = ds["ua"].mean("lon")   # zonal mean; must end up as (lev, lat)

    fig  = FigurePanel(figsize=(8, 5))
    plot = LatHgtPlot(fig, subplot_pos=(1, 1, 1))

    (plot
        .add_shade(data, vmax=15, vnum=17, color="RdBu_r", withLine=True)
        .add_gridlines()
        .add_colorbar()
        .add_title(title="Zonal-Mean Zonal Wind Anomaly")
        .add_corner(left_text="Sample", right_text="[DJF]"))

    fig.render(ofile="lathgt_sample.png")
    print("Saved lathgt_sample.png")


if __name__ == "__main__":
    main()
