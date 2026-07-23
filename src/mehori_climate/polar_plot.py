import numpy as np
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs

# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import cartopy.feature as cfeature
# from   cartopy.util import add_cyclic_point
from   cartopy.mpl.gridliner import LONGITUDE_FORMATTER


from mehori_climate import LatlonPlot

class NpsPlot(LatlonPlot):
    """
    North Polar Stereographic map. Extends LatlonPlot.

    Unlike the old version, this now takes `fig`/`subplot_pos` (instead of
    building its own figure) and calls LatlonPlot.__init__() via super(),
    so it inherits fontscale, zorder/levels/cmap_name state, and aspect
    handling -- meaning add_shade(), add_colorbar(), add_title(),
    add_corner(), add_contour(), add_vector(), etc. all work unmodified,
    and it's compatible with FigurePanel/multi-panel layouts just like
    LatlonPlot:

        fig    = FigurePanel(figsize=(8, 8))
        polar1 = NpsPlot(fig, subplot_pos=(1, 1, 1))
        polar1.add_shade(data).add_coastlines().add_gridlines().add_colorbar()
        fig.render(ofile="polar_map.png")

    Only the projection, polar extent, and circular map boundary are
    specific to this subclass.
    """
    def __init__(self, fig, subplot_pos=(1, 1, 1), font_family='', central_longitude=180,
                 extent=(-180, 180, 20, 90), fontscale=None, projection=None):
        # projection is overridable so subclasses (e.g. SpsPlot) can swap in
        # SouthPolarStereo while still reusing all of this __init__.
        if projection is None:
            projection = ccrs.NorthPolarStereo(central_longitude=central_longitude)
        super().__init__(
            fig, subplot_pos=subplot_pos, font_family=font_family,
            central_longitude=central_longitude, aspect='equal', fontscale=fontscale,
            projection=projection,
        )

        self.ax.set_extent(list(extent), crs=ccrs.PlateCarree())

        # Force the map boundary to be a perfect circle (Pro-Tip)
        theta  = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        self.ax.set_boundary(circle, transform=self.ax.transAxes)

    def add_gridlines(self, labelsize=10, ylocator=(20, 40, 60, 80), meridian_lat_max=80):
        """
        Overrides LatlonPlot.add_gridlines() for the polar-specific style.
        ylocator: latitude circles to draw. Defaults to stopping at 80 --
        without an explicit locator, cartopy's auto-picked latitudes can
        include one right at the pole (90), which draws as a degenerate
        circle collapsed to a point at the plot's center.
        meridian_lat_max: latitude the longitude "spoke" lines stop at.
        cartopy draws meridians across the whole axes extent by default,
        so even with ylocator capped at 80 they'd still run in to the pole
        at the plot's center. Set to None to restore that default behavior.
        """
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                                linestyle='--', rotate_labels=False, xpadding=7)
        gl.xlabel_style = {'size': labelsize * self.fontscale, 'rotation': 0}
        xlons = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
        gl.xlocator = mticker.FixedLocator(xlons)
        gl.ylocator = mticker.FixedLocator(list(ylocator))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = mticker.NullFormatter()

        if meridian_lat_max is not None:
            # Turn off cartopy's own full-extent meridian lines (labels
            # stay on) and draw truncated ones instead, stopping short of
            # the pole to match ylocator's latitude-circle cutoff.
            gl.xlines = False
            lat_min = self.ax.get_extent(ccrs.PlateCarree())[2]
            lats = np.linspace(lat_min, meridian_lat_max, 30)
            for lon_val in xlons:
                self.ax.plot(np.full_like(lats, lon_val), lats, transform=ccrs.PlateCarree(),
                             color='gray', alpha=0.5, linestyle='--', linewidth=0.5,
                             zorder=self.zorder)
            self.zorder += 1

        return self

    def add_title(self, title="", pad=40, fontsize=14):
        """
        Overrides LatlonPlot.add_title(): on this circular polar boundary,
        the gridliner's "0 deg" longitude label sits right at the top of
        the circle, so the title needs more clearance than the flat-map
        default (pad=5) to avoid overlapping it.
        """
        return super().add_title(title=title, pad=pad, fontsize=fontsize)


class SpsPlot(NpsPlot):
    """
    South Polar Stereographic map. Extends NpsPlot, reusing everything --
    fontscale, zorder/levels/cmap_name state, circular boundary, title
    clearance -- just swapping the projection to SouthPolarStereo and
    flipping the default extent/gridline latitudes to the southern
    hemisphere.

        fig    = FigurePanel(figsize=(8, 8))
        polar1 = SpsPlot(fig, subplot_pos=(1, 1, 1))
        polar1.add_shade(data).add_coastlines().add_gridlines().add_colorbar()
        fig.render(ofile="south_polar_map.png")
    """
    def __init__(self, fig, subplot_pos=(1, 1, 1), font_family='', central_longitude=0,
                 extent=(0, 360, -90, -20), fontscale=None):
        super().__init__(
            fig, subplot_pos=subplot_pos, font_family=font_family,
            central_longitude=central_longitude, extent=extent, fontscale=fontscale,
            projection=ccrs.SouthPolarStereo(central_longitude=central_longitude),
        )

    def add_gridlines(self, labelsize=14, ylocator=(-20, -40, -60, -80)):
        """ Overrides NpsPlot.add_gridlines(): southern-hemisphere latitude circles. """
        return super().add_gridlines(labelsize=labelsize, ylocator=ylocator)
