import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath

# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from   cartopy.util import add_cyclic_point
from   cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from mehori_tools import LatlonPlot

class npsPlot(LatlonPlot):
    """
    An Object-Oriented builder for plots.
    Extends LatlonPlot
    """
    def __init__(self, figsize=(12, 8), central_longitude=180, font_family='Nimbus Sans'):
        # 0. Set default font
        plt.rcParams['font.family'] = [font_family]
        self.transform = ccrs.PlateCarree()
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=central_longitude))
        self.ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())

        # Force the map to be a perfect circle (Pro-Tip)
        self.theta = np.linspace(0, 2*np.pi, 100)
        self.center, self.radius = [0.5, 0.5], 0.5
        self.verts = np.vstack([np.sin(self.theta), np.cos(self.theta)]).T
        self.circle = mpath.Path(self.verts * self.radius + self.center)
        self.ax.set_boundary(self.circle, transform=self.ax.transAxes)

    def add_gridlines(self):
        self.gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', rotate_labels=False, xpadding=7)
        self.gl.xlabel_style = {'size': 14, 'rotation': 0}
        self.gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
        self.gl.xformatter = LONGITUDE_FORMATTER
        self.gl.yformatter = mticker.NullFormatter()
        return self


