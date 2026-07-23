import matplotlib.pyplot as plt

class FigurePanel:
    """
    Thin wrapper around a matplotlib Figure for multi-panel layouts.

    Drop this in anywhere a LatlonPlot expects `fig` -- any attribute not
    defined here (add_subplot, add_axes, colorbar, canvas, savefig, ...)
    is forwarded straight to the underlying Figure via __getattr__, so
    LatlonPlot needs no changes to work with it.

    The one thing it adds is render(): figure-level finalization (margins,
    save/show) lives on the figure itself and runs exactly once, no matter
    how many panels are attached -- instead of each panel's own render()
    call reflowing the shared GridSpec and silently invalidating layout
    decisions (like colorbar placement) already made by sibling panels.

    The real matplotlib Figure is stored as `.figure`, not `.fig` --
    LatlonPlot stores whatever you pass it (this FigurePanel, or a raw
    Figure) as `plot.fig`, so naming this attribute `.fig` too would mean
    `plot.fig.fig` to reach the real Figure from a plot instance. You
    normally don't need `.figure` directly anyway: any Figure method
    (`.axes`, `.savefig`, ...) is already reachable straight off the
    FigurePanel itself via __getattr__.

    Usage:
        fig   = FigurePanel(figsize=(10, 6))
        plot1 = LatlonPlot(fig, subplot_pos=(1, 2, 1))
        plot2 = LatlonPlot(fig, subplot_pos=(1, 2, 2))
        # ... build both panels fully, including add_colorbar() on each ...
        fig.render(ofile="panel_map.png")
    """
    def __init__(self, figsize=(10, 6), **kwargs):
        self.figure = plt.figure(figsize=figsize, **kwargs)

    def __getattr__(self, name):
        # Only reached for attributes not found on FigurePanel itself, so
        # this transparently proxies to the wrapped Figure.
        return getattr(self.figure, name)

    def render(self, ofile="", bottom=0.15, close_fig=False):
        """ Finalizes layout and either saves or displays the figure. """
        self.figure.subplots_adjust(bottom=bottom)

        if ofile:
            # NOTE: bbox_inches="tight" is deliberately NOT used here. It
            # has a known bad interaction with cartopy's gridline label
            # layout (auto_update runs the label-positioning pass twice),
            # which crops out most of the map and leaves only a sliver plus
            # any plain (non-geo) axes like a colorbar. subplots_adjust
            # above handles margins instead.
            self.figure.savefig(ofile, dpi=300, transparent=False)
        else:
            plt.figure(self.figure.number)  # ensure self.figure is the active figure
            plt.show()

        if close_fig:
            plt.close(self.figure)
        return self
