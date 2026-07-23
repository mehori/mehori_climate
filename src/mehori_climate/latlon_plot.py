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
    Assumes inputs are xarray DataArrays with a latitude and a longitude
    dimension (auto-detected from common names -- see _lat_name/_lon_name
    -- so 'lat'/'lon' or 'latitude'/'longitude' both work).
    """

    # Candidate dimension names, checked in order, for auto-detecting the
    # lat/lon dims on any xarray DataArray passed in. Subclasses can add
    # their own candidate lists the same way (e.g. LatHgtPlot's
    # _LEV_DIM_CANDIDATES for the vertical dimension) and reuse _detect_dim.
    _LAT_DIM_CANDIDATES = ('lat', 'latitude')
    _LON_DIM_CANDIDATES = ('lon', 'longitude')

    def __init__(self, fig, subplot_pos=(1, 1, 1), font_family='', central_longitude=180, aspect="", fontscale=None, projection=None, extent=None):
        # 0. Set default font
        if font_family:
            plt.rcParams['font.family'] = [font_family]

        # Setup Figure and Axes
        self.fig        = fig
        # Subclasses (e.g. NpsPlot) can pass a different cartopy projection
        # for self.ax while still reusing all of this __init__ (fontscale,
        # zorder, levels/cmap_name state, aspect handling, etc.) via super().
        # projection=False opts out of cartopy entirely (plain matplotlib
        # Axes) for non-map subclasses like LatHgtPlot.
        if projection is False:
            self.projection = None
            self.transform  = None
            self.ax         = fig.add_subplot(*subplot_pos)
        else:
            self.projection = projection if projection is not None else ccrs.PlateCarree(central_longitude=central_longitude)
            self.transform  = ccrs.PlateCarree()
            self.ax         = fig.add_subplot(*subplot_pos, projection=self.projection)

        # Set aspect ratio (plain Cartesian axes default to 'auto' instead
        # of 'equal' -- lat vs. pressure/height have unrelated units/scales)
        if aspect:
            self.ax.set_aspect(aspect)
        elif projection is not False:
            self.ax.set_aspect('equal')

        # Font scaling: text is set in absolute points, so stacking panels
        # into a grid (more rows/cols -> smaller panels) makes text look
        # relatively larger unless scaled down. Auto-derive from the
        # subplot grid unless the caller passes an explicit value.
        if fontscale is None:
            nrows, ncols = subplot_pos[0], subplot_pos[1]
            fontscale = 1.0 / max(nrows, ncols) ** 0.5
        self.fontscale  = fontscale

        # State variables for colorbar
        self.levels     = None
        self.cmap_name  = None
        self.zorder     = 1

        # Without an explicit extent, cartopy shows the full global view
        # for the projection regardless of how much of the data actually
        # has values -- add_shade() only draws data, it doesn't shrink the
        # view to match it. Optional here so it can be set at construction
        # time; set_extent() below covers setting/changing it afterward.
        if extent is not None and projection is not False:
            self.set_extent(extent)

    def _detect_dim(self, data, candidates, kind):
        """
        Generic dimension-name auto-detection: returns the first name in
        `candidates` that's actually present in data.dims. Used for lat/lon
        here, and reused by subclasses (e.g. LatHgtPlot's vertical
        pressure dimension) with their own candidate list.
        """
        for name in candidates:
            if name in data.dims:
                return name
        raise ValueError(
            f"Could not auto-detect a {kind} dimension in dims={data.dims}; "
            f"expected one of {candidates}."
        )

    def _lat_name(self, data):
        return self._detect_dim(data, self._LAT_DIM_CANDIDATES, "latitude")

    def _lon_name(self, data):
        return self._detect_dim(data, self._LON_DIM_CANDIDATES, "longitude")

    def add_cyclic(self, data):
        """Helper to add cyclic point to avoid the white line at 0 degrees."""
        data_c, lon_c = add_cyclic_point(data.values, coord=data[self._lon_name(data)])
        return data_c, lon_c

    def _cyclic_data(self, data, cyclic=True):
        """
        Shared "add a cyclic (wraparound) longitude point, or don't" branch
        used by add_shade()/add_contour(). Returns (data_2d, lon).
        """
        if cyclic:
            return self.add_cyclic(data)
        return data, data[self._lon_name(data)]

    def set_extent(self, extent, crs=None):
        """
        Restricts the visible map to [lon_min, lon_max, lat_min, lat_max]
        (PlateCarree degrees by default). Chainable, so it can be called
        any time after construction -- e.g. once the data's actual lat/lon
        range is known:

            plot1.add_shade(data).set_extent([-180, 180, 0, 90])
        """
        self.ax.set_extent(list(extent), crs=crs or self.transform)
        return self

    def add_coastlines(self, fillland=False):
        if fillland:
            land  = cfeature.NaturalEarthFeature('physical', 'land', "50m", edgecolor='face', facecolor="lightgray", zorder=self.zorder)
            self.ax.add_feature(land)
        else:
            self.ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#222222', linewidth=0.8)
            # self.ax.coastlines(resolution='50m', color='black', linewidth=1, facecolor="lightgray" )
        self.zorder += 1
        return self

    def _make_levels(self, vmin, vmax, vnum, symmetric, interval=None):
        # explicit interval is given
        if interval is not None:
            if symmetric:
                vabs = max(abs(vmin), abs(vmax))
                n = int(round(vabs / interval))
                levels_pos = np.arange(n + 1) * interval
                levels_neg = -levels_pos[::-1]
                return np.unique(np.concatenate([levels_neg, levels_pos]))
            n = int(round((vmax - vmin) / interval))
            return vmin + np.arange(n + 1) * interval

        # no interval but symmetric
        if symmetric:
            half = vnum // 2
            levels_neg = np.linspace(vmin, 0, half + 1)
            levels_pos = np.linspace(0, vmax, half + 1)
            return np.unique(np.concatenate([levels_neg, levels_pos]))

        # no interval and no symmetric
        return np.linspace(vmin, vmax, vnum)

    def _resolve_vmin_vmax(self, dmin, dmax, vmin, vmax, symmetric):
        """
        Shared vmin/vmax/symmetric resolution logic for add_shade(),
        used by both LatlonPlot and LatHgtPlot (inherited):
        - vmin and vmax both given: explicit range, not symmetric unless
          forced via `symmetric`.
        - only vmax given: legacy anomaly-style call, symmetric about zero
          (vmin = -vmax) unless forced otherwise.
        - only vmin given: explicit lower bound, upper bound taken from
          the data's own max.
        - neither given: auto-range from the data; symmetric about zero
          if the data straddles zero, otherwise the data's own min/max.
        Returns (vmin, vmax, symmetric).
        """
        if vmin is not None and vmax is not None:
            symmetric = False if symmetric is None else symmetric
        elif vmax is not None:
            symmetric = True if symmetric is None else symmetric
            vmin = -vmax
        elif vmin is not None:
            symmetric = False if symmetric is None else symmetric
            vmax = dmax
        else:
            symmetric = (dmin < 0 < dmax) if symmetric is None else symmetric
            if symmetric:
                vabs = max(abs(dmin), abs(dmax))
                vmin, vmax = -vabs, vabs
            else:
                vmin, vmax = dmin, dmax
        return vmin, vmax, symmetric

    def add_shade(self, data, vmin=None, vmax=None, vnum=17, interval=None,
              color="RdBu_r", linecolor="#444", cyclic=True, withLine=False,
              alpha=1.0, symmetric=None):
        """Adds a filled contour shading layer.
           vmin/vmax: explicit bounds. Leave both unset to auto-range from data.
           Setting only vmax keeps old anomaly behavior (symmetric, vmin=-vmax).
           vnum: number of levels (ignored if interval is given).
           interval: fixed step between levels, e.g. interval=5 for 990,995,...,1030.
           symmetric: force/override symmetric-about-zero levels; auto-inferred
           if not given.
        """

        # if cyclic, add 360 point
        lat_name = self._lat_name(data)
        data_c, lon_c = self._cyclic_data(data, cyclic)

        # get data max/min
        dmin = float(np.nanmin(data_c))
        dmax = float(np.nanmax(data_c))

        # set shading levels
        vmin, vmax, symmetric = self._resolve_vmin_vmax(dmin, dmax, vmin, vmax, symmetric)
        self.levels = self._make_levels(vmin, vmax, vnum, symmetric, interval)
        self.cmap_name = color

        # draw shading
        self.ax.contourf(
            lon_c, data[lat_name], data_c,
            levels=self.levels,
            transform=self.transform,
            cmap=self.cmap_name,
            extend='both',
            alpha=alpha,
            zorder=self.zorder
        )
        self.zorder += 1

        # draw line on top
        if withLine:
            self.ax.contour(
                lon_c, data[lat_name], data_c,
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
        lat_name = self._lat_name(pval)
        pval_c, lon_c = self.add_cyclic(pval)
        # matplotlib has no per-call hatch color, only the global rcParams,
        # so save/restore around the draw instead of overwriting it
        # permanently (which would otherwise leak into any other hatched
        # plot drawn later in the same session).
        prev_hatch_color     = mpl.rcParams['hatch.color']
        prev_hatch_linewidth = mpl.rcParams['hatch.linewidth']
        try:
            mpl.rcParams['hatch.color'] = 'gray'
            mpl.rcParams['hatch.linewidth'] = 0.5
            self.ax.contourf(
                lon_c, pval[lat_name], pval_c,
                levels=[0, confidence, 1],
                hatches=['..', None],
                colors='none',
                transform=self.transform
            )
        finally:
            mpl.rcParams['hatch.color']     = prev_hatch_color
            mpl.rcParams['hatch.linewidth'] = prev_hatch_linewidth
        return self

    def add_contour(self, data, levels=10, colors="black", linewidths=1.0, cyclic=True):
        """Extension: Adds a line contour layer."""
        lat_name = self._lat_name(data)
        data_c, lon_c = self._cyclic_data(data, cyclic)

        self.ax.contour(
            lon_c, data[lat_name], data_c,
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
        # Slicing with [::skip] avoids overcrowding the map with arrows.
        # cartopy's quiver() does raw numpy-style boolean-mask indexing
        # internally while transforming vectors; xarray DataArrays don't
        # support that indexing style and raise an IndexError, so pass
        # plain numpy arrays via .values instead.
        lon = u_data[self._lon_name(u_data)].values[::skip]
        lat = u_data[self._lat_name(u_data)].values[::skip]
        u   = u_data.values[::skip, ::skip]
        v   = v_data.values[::skip, ::skip]

        quiver_style = dict(
            width=0.0018,           # 矢軸の太さ
            headwidth=3,           # 矢頭の幅（矢軸の太さに対する倍率）
            headlength=5,          # 矢頭の長さ（矢軸の太さに対する倍率）
            headaxislength=5,    # 矢頭の根本部分の長さ
            pivot='tail'           # 矢印の中心を座標に合わせる
        )

        self.ax.quiver(
            lon, lat, u, v,
            facecolor="black",
            alpha=0.7,
            transform=self.transform,
            scale=scale,
            zorder=self.zorder,
            **quiver_style
        )
        self.zorder += 1
        return self

    def add_gridlines(self, labelsize=10):
        """ Adds latitude and longitude gridlines.  """
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels   = False
        gl.right_labels = False
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': labelsize * self.fontscale}
        gl.ylabel_style = {'size': labelsize * self.fontscale}
        return self

    def add_colorbar(self, height=0.02, pad=0.06, width_frac=0.7, labelsize=10, rect=None):
        """
        Adds a horizontal colorbar below this panel's own axes.
        By default the position/size is derived from this panel's own
        `self.ax` (via width_frac/height/pad, all in figure-fraction units
        scaled by self.fontscale), so each LatlonPlot instance places its
        own colorbar under itself automatically -- panels can be stacked,
        arranged in a grid, or resized without recalculating any x/y
        coordinates by hand. Pass an explicit `rect=[x0, y0, w, h]`
        (figure-fraction) to override this and position it manually.
        """
        if self.levels is None or self.cmap_name is None:
            raise ValueError("Cannot add colorbar: Call add_shade() first to set levels and colormap.")

        if rect is None:
            self.fig.canvas.draw()  # finalize layout (aspect) before reading position
            pos = self.ax.get_position()
            h  = height * self.fontscale
            p  = pad * self.fontscale
            cb_width = pos.width * width_frac
            x0 = pos.x0 + (pos.width - cb_width) / 2
            y0 = pos.y0 - p
            rect = [x0, y0, cb_width, h]

        cmap = plt.get_cmap(self.cmap_name)
        norm = mcolors.BoundaryNorm(self.levels, ncolors=cmap.N)
        fake_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        fake_mappable.set_array([])

        cbar_ax = self.fig.add_axes(rect)
        cbar = self.fig.colorbar(fake_mappable, cax=cbar_ax, orientation='horizontal', extend='neither')
        cbar.set_ticks(self.levels[::2])
        cbar.ax.tick_params(labelsize=labelsize * self.fontscale)
        return self

    def add_corner(self, left_text="", right_text="", fontsize=14):
        """Adds title and corner text."""
        if left_text:
            self.ax.text(0.00, 1.010, left_text, transform=self.ax.transAxes,
                         fontsize=fontsize * self.fontscale, va='bottom', ha='left', clip_on=False)
        if right_text:
            self.ax.text(1.00, 1.010, right_text, transform=self.ax.transAxes,
                         fontsize=fontsize * self.fontscale, va='bottom', ha='right', clip_on=False)
        return self

    def add_title(self, title="", pad=5, fontsize=14):
        if title:
            # Uses ax.text() instead of ax.set_title(): on this cartopy
            # GeoAxes, the dedicated Title artist from set_title() doesn't
            # reliably get drawn (cartopy's GeoAxes.draw() override can
            # skip it). add_corner() uses this same ax.text() approach and
            # renders correctly, so title mirrors it for consistency.
            y = 1.0 + pad / 500.0
            self.ax.text(0.5, y, title, transform=self.ax.transAxes,
                         fontsize=fontsize * self.fontscale, va='bottom', ha='center', clip_on=False)
        return self

    def make_clevel_cint(self, data, cint=1):
        min_val = np.floor(np.min(data))
        max_val = np.floor(np.max(data))
        levels = np.arange(min_val, max_val + cint, cint)
        return levels


