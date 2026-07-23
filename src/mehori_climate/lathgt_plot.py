import numpy as np

from mehori_climate import LatlonPlot


class LatHgtPlot(LatlonPlot):
    """
    Latitude-height (pressure cross-section) plot. Extends LatlonPlot.

    Unlike LatlonPlot/NpsPlot/SpsPlot this isn't a map -- there's no
    cartopy projection, just a plain matplotlib Axes with latitude on x
    and a log-pressure height coordinate on y. It's built with
    LatlonPlot.__init__(..., projection=False), so it still inherits
    fontscale, zorder/levels/cmap_name state, and aspect handling, which
    means add_colorbar(), add_title(), and add_corner() all work
    unmodified (they only ever touch self.ax/self.fig generically). Only
    the data-plotting methods (add_shade, add_contour) and gridlines are
    overridden, since the parent's versions assume a GeoAxes.

    Height is computed as -log(pressure / 1000): a standard log-pressure
    height coordinate, evenly spaced in log-pressure, with the y-axis
    tick labels showing actual hPa values (surface at the bottom).

        fig  = FigurePanel(figsize=(8, 5))
        plot = LatHgtPlot(fig, subplot_pos=(1, 1, 1))
        plot.add_shade(data, vnum=17, color="RdBu_r").add_gridlines().add_colorbar()
        fig.render(ofile="lathgt_map.png")

    `data` passed to add_shade()/add_contour() must be a 2D xarray
    DataArray with a 'lat' dim and a vertical pressure dim (auto-detected
    from {'level', 'lev', 'plev', 'pressure', 'pres'} unless given
    explicitly via `lev_dim`).
    """

    # Candidate names for the vertical pressure dimension, checked in
    # order when lev_dim isn't given explicitly.
    _LEV_DIM_CANDIDATES = ('level', 'lev', 'plev', 'pressure', 'pres')

    # Readable pressure levels (hPa) to consider for y-axis ticks.
    _PRESSURE_TICK_CANDIDATES = np.array(
        [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1]
    )

    def __init__(self, fig, subplot_pos=(1, 1, 1), font_family='', aspect='auto', fontscale=None):
        super().__init__(
            fig, subplot_pos=subplot_pos, font_family=font_family,
            aspect=aspect, fontscale=fontscale, projection=False,
        )
        self._yaxis_set = False

    @staticmethod
    def _log_height(pressure):
        """ log-pressure height coordinate: -log(p / 1000 hPa). """
        return -np.log(np.asarray(pressure, dtype=float) / 1000.0)

    def _detect_lev_dim(self, data):
        for name in self._LEV_DIM_CANDIDATES:
            if name in data.dims:
                return name
        raise ValueError(
            f"Could not auto-detect a vertical pressure dimension in dims={data.dims}; "
            f"pass lev_dim explicitly."
        )

    def _setup_axes(self, pressure, lat):
        """ Sets up the y-axis (log-pressure height, hPa tick labels) and
        x-axis (latitude) once, based on whichever call to add_shade()/
        add_contour() runs first -- ranges are derived from the actual
        data passed in, not hardcoded, so the view follows however the
        data was sliced (e.g. lat=0-90 only shows 0-90). Does nothing if
        set_extent() was already called explicitly. """
        if self._yaxis_set:
            return

        p = np.asarray(pressure, dtype=float)
        pmin, pmax = float(p.min()), float(p.max())

        ticks = self._PRESSURE_TICK_CANDIDATES
        ticks = ticks[(ticks >= pmin) & (ticks <= pmax)]
        if len(ticks) < 2:
            ticks = np.sort(np.unique(p))[::-1]

        self.ax.set_yticks(self._log_height(ticks))
        self.ax.set_yticklabels([str(int(t)) for t in ticks])
        self.ax.set_ylim(self._log_height(pmax), self._log_height(pmin))  # surface (high p) at bottom
        self.ax.set_ylabel("Pressure (hPa)")

        lat = np.asarray(lat, dtype=float)
        self._set_lat_axis(float(lat.min()), float(lat.max()))

        self._yaxis_set = True

    def _set_lat_axis(self, lat_min, lat_max, step=30):
        self.ax.set_xlim(lat_min, lat_max)
        ticks = np.arange(np.ceil(lat_min / step) * step, lat_max + 1e-9, step)
        if len(ticks) < 2:
            ticks = np.linspace(lat_min, lat_max, 5)
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(
            [f"{abs(t):.0f}°S" if t < 0 else (f"{t:.0f}°N" if t > 0 else "0°") for t in ticks]
        )
        self.ax.set_xlabel("Latitude")

    def set_extent(self, lat=None, plev=None):
        """
        Overrides LatlonPlot.set_extent(): this is a plain (non-cartopy)
        Axes, so there's no ax.set_extent() to call -- `lat` sets xlim and
        `plev` sets ylim (converted to log-pressure height) instead.
        Chainable, and safe to call before add_shade() (to pre-set the
        view) or after (to override the range auto-derived from the data):

            plot.add_shade(data).set_extent(lat=[0, 90], plev=[1000, 100])

        Either argument can be omitted to leave that axis alone.
        """
        if lat is not None:
            self._set_lat_axis(min(lat), max(lat))

        if plev is not None:
            pmin, pmax = min(plev), max(plev)
            ticks = self._PRESSURE_TICK_CANDIDATES
            ticks = ticks[(ticks >= pmin) & (ticks <= pmax)]
            if len(ticks) < 2:
                ticks = np.array([pmin, pmax])
            self.ax.set_yticks(self._log_height(ticks))
            self.ax.set_yticklabels([str(int(t)) for t in ticks])
            self.ax.set_ylim(self._log_height(pmax), self._log_height(pmin))
            self.ax.set_ylabel("Pressure (hPa)")

        self._yaxis_set = True
        return self

    def add_shade(self, data, lev_dim=None, vmin=None, vmax=None, vnum=17, interval=None,
                  color="RdBu_r", linecolor="#444", withLine=False, alpha=1.0, symmetric=None):
        """ Adds filled contour shading. See class docstring for `data`/`lev_dim`. """
        lev_dim  = lev_dim or self._detect_lev_dim(data)
        pressure = data[lev_dim].values
        height   = self._log_height(pressure)
        values   = data.transpose(lev_dim, 'lat').values

        dmin = float(np.nanmin(values))
        dmax = float(np.nanmax(values))

        # Same level-selection logic as LatlonPlot.add_shade().
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

        self.levels    = self._make_levels(vmin, vmax, vnum, symmetric, interval)
        self.cmap_name = color

        self.ax.contourf(
            data.lat, height, values,
            levels=self.levels, cmap=self.cmap_name, extend='both',
            alpha=alpha, zorder=self.zorder,
        )
        self.zorder += 1

        if withLine:
            self.ax.contour(
                data.lat, height, values,
                levels=self.levels, colors=linecolor, linewidths=0.5,
                zorder=self.zorder,
            )
        self.zorder += 1

        self._setup_axes(pressure, data.lat.values)
        return self

    def add_contour(self, data, lev_dim=None, levels=10, colors="black", linewidths=1.0):
        """ Overrides LatlonPlot.add_contour(): plain (non-cartopy) line contours. """
        lev_dim  = lev_dim or self._detect_lev_dim(data)
        pressure = data[lev_dim].values
        height   = self._log_height(pressure)
        values   = data.transpose(lev_dim, 'lat').values

        self.ax.contour(
            data.lat, height, values,
            levels=levels, colors=colors, alpha=0.6, linewidths=linewidths,
            zorder=self.zorder,
        )
        self.zorder += 1

        self._setup_axes(pressure, data.lat.values)
        return self

    def add_gridlines(self):
        """ Overrides LatlonPlot.add_gridlines(): plain matplotlib grid (no cartopy Gridliner). """
        self.ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        return self
