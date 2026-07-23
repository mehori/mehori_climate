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

    Usage:
        fig   = FigurePanel(figsize=(10, 6))
        plot1 = LatlonPlot(fig, subplot_pos=(1, 2, 1))
        plot2 = LatlonPlot(fig, subplot_pos=(1, 2, 2))
        # ... build both panels fully, including add_colorbar() on each ...
        fig.render(ofile="panel_map.png")
    """
    def __init__(self, figsize=(10, 6), **kwargs):
        self.fig = plt.figure(figsize=figsize, **kwargs)

    def __getattr__(self, name):
        # Only reached for attributes not found on FigurePanel itself, so
        # this transparently proxies to the wrapped Figure.
        return getattr(self.fig, name)

    def render(self, ofile="", bottom=0.15, close_fig=False):
        """ Finalizes layout and either saves or displays the figure. """
        self.fig.subplots_adjust(bottom=bottom)

        if ofile:
            self.fig.savefig(ofile, dpi=300, transparent=False)
        else:
            plt.figure(self.fig.number)  # ensure self.fig is the active figure
            plt.show()

        if close_fig:
            plt.close(self.fig)
        return self
