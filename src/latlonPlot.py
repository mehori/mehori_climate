import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from   cartopy.util import add_cyclic_point
from   cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

class LatlonPlot:
    """
    An Object-Oriented builder for plots.
    Assumes inputs are xarray DataArrays with 'lat' and 'lon' coordinates.
    """
    def __init__(self, fig, subplot_pos=(1, 1, 1), central_longitude=180, font_family='Nimbus Sans', aspect=""):
        # 0. Set default font
        plt.rcParams['font.family'] = [font_family]
        
        # Setup Figure and Axes
        self.fig        = fig
        self.projection = ccrs.PlateCarree(central_longitude=central_longitude)
        self.transform  = ccrs.PlateCarree()
        self.ax         = fig.add_subplot(*subplot_pos, projection=self.projection)

        # Set aspect ratio
        if aspect:
            self.ax.set_aspect(aspect)
        else:
            self.ax.set_aspect('equal')
        
        # State variables for colorbar
        self.levels     = None
        self.cmap_name  = None
        self.zorder     = 1

    def add_cyclic(self, data):
        """Helper to add cyclic point to avoid the white line at 0 degrees."""
        data_c, lon_c = add_cyclic_point(data.values, coord=data.lon)
        return data_c, lon_c

    def add_coastlines(self, fillland=False):
        if fillland:
            land  = cfeature.NaturalEarthFeature('physical', 'land', "50m", edgecolor='face', facecolor="lightgray", zorder=self.zorder)
            self.ax.add_feature(land)
        else:
            self.ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#222222', linewidth=0.8)
            # self.ax.coastlines(resolution='50m', color='black', linewidth=1, facecolor="lightgray" )
        self.zorder += 1
        return self

    def add_shade(self, data, vmax=5, vnum=17, color="RdBu_r", linecolor="#444", cyclic=True, withLine=False, alpha=1.0):
        """Adds a filled contour shading layer."""

        if cyclic:
            data_c, lon_c = self.add_cyclic(data)
        else:
            data_c, lon_c = data, data.lon

        half_levels = int(vnum / 2)
        levels_neg = np.linspace(-vmax,    0, half_levels )
        levels_pos = np.linspace(    0, vmax, half_levels )
        self.levels = np.unique(np.concatenate((levels_neg, levels_pos)))
        self.cmap_name = color
        self.ax.contourf(
            lon_c, data.lat, data_c,
            levels=self.levels,
            transform=self.transform,
            cmap=self.cmap_name,
            extend='both',
            alpha=alpha,
            zorder=self.zorder
        )
        self.zorder += 1

        # overlay contours
        if withLine:
            self.ax.contour(
                lon_c, data.lat, data_c,
                levels=self.levels,
                colors=linecolor,
                linewidths=0.5,
                transform=ccrs.PlateCarree(),
                zorder=self.zorder 
            )
        self.zorder += 1
        return self

    def add_significance_dots(self, pval, confidence=0.05):
        """
        given a p-value array overlays gray significance dots. default is 2-tailed 95% 
        """
        pval_c, lon_c = self.add_cyclic(pval)
        mpl.rcParams['hatch.color'] = 'gray'
        mpl.rcParams['hatch.linewidth'] = 0.5
        self.ax.contourf(
            lon_c, pval.lat, pval_c,
            levels=[0, confidence, 1],
            hatches=['..', None],
            colors='none',
            transform=self.transform
        )
        return self

    def add_contour(self, data, levels=10, colors="black", linewidths=1.0, cyclic=True):
        """Extension: Adds a line contour layer."""
        if cyclic:
            data_c, lon_c = self.add_cyclic(data)
        else:
            data_c, lon_c = data, data.lon
        print(self.levels)

        self.ax.contour(
            lon_c, data.lat, data_c,
            levels=levels,
            colors=colors,
            alpha=0.6,
            linewidths=linewidths,
            transform=self.transform,
            zorder=self.zorder
        )
        self.zorder += 1
        return self

    def add_vector(self, u_data, v_data, skip=2, scale=None):
        """Extension: Adds wind/current vector arrows using quiver."""
        # Slicing with [::skip] avoids overcrowding the map with arrows
        self.ax.quiver(
            lon_c[::skip], u_data.lat[::skip], 
            u_c[::skip, ::skip], v_c[::skip, ::skip],
            transform=self.transform,
            scale=scale
        )
        return self

    def add_gridlines(self):
        """ Adds latitude and longitude gridlines.  """
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels   = False
        gl.right_labels = False
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        return self

    def add_colorbar(self, rect=[0.263, 0.17, 0.5, 0.02]):
        """
        Adds a horizontal colorbar at the bottom.
        """
        if self.levels is None or self.cmap_name is None:
            raise ValueError("Cannot add colorbar: Call add_shade() first to set levels and colormap.")
            
        cmap = plt.get_cmap(self.cmap_name)
        norm = mcolors.BoundaryNorm(self.levels, ncolors=cmap.N)
        fake_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        fake_mappable.set_array([])

        cbar_ax = self.fig.add_axes(rect)
        cbar = self.fig.colorbar(fake_mappable, cax=cbar_ax, orientation='horizontal', extend='neither')
        cbar.set_ticks(self.levels[::2])
        return self

    def add_corner(self, left_text="", right_text="", fontsize=16):
        """Adds title and corner text."""
        if left_text:
            self.ax.text(0.00, 1.010, left_text, transform=self.ax.transAxes,
                         fontsize=fontsize, va='bottom', ha='left', clip_on=False)
        if right_text:
            self.ax.text(1.00, 1.010, right_text, transform=self.ax.transAxes,
                         fontsize=fontsize, va='bottom', ha='right', clip_on=False)
        return self

    def add_title(self, title="", pad=0, fontsize=14):
        if title:
            self.ax.set_title(title, pad=pad, fontsize=fontsize)
        return self

    def make_clevel_cint(self, data, cint=1):
        min_val = np.floor(np.min(data))
        max_val = np.floor(np.max(data))
        levels = np.arange(min_val, max_val + cint, cint)
        return levels

    def render(self, ofile="", close_fig=False):
        """ Finalizes layout and either saves or displays the figure.  """
        plt.subplots_adjust(bottom=0.15) # Ensure colorbar doesn't clip
        if ofile:
            plt.savefig(ofile, dpi=300, bbox_inches="tight", transparent=False)
        else:
            plt.show()

        # if close_fig is True, memory is cleared
        if close_fig:
            plt.close(self.fig)  
