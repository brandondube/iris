"""Tools for plotting the results of wavefront sensing."""
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator

from prysm.util import share_fig_ax
from prysm.mathops import sqrt as _sqrt


def _plot_attribute_global(nested_iterable, ax):
    xmin = 0
    for iteration in nested_iterable:
        len_ = len(iteration)
        x = range(xmin, xmin + len_)
        ax.plot(x, iteration)
        xmin += len_

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_costfunction_history_global(document, sqrt=False, fig=None, ax=None):
    """Plot the cost function history of a global optimization run.

    Parameters
    ----------
    document : `dict`
        document produced by utilities.prepare_document_global
    sqrt : `bool`
        whether to take the sqrt of the cost function
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    fig, ax = share_fig_ax(fig, ax)
    if sqrt:
        data = [[_sqrt(x) for x in y] for y in document['cost_iter']]
        label = r'$\sqrt{Cost\, Function\,}$ [a.u.]'
    else:
        data = document['cost_iter']
        label = 'Cost Function [a.u.]'
    _plot_attribute_global(data, ax=ax)
    ax.set(ylabel=label)
    return fig, ax


def plot_rrmswfe_history_global(document, fig=None, ax=None):
    """Plot the residual RMS WFE history of a global optimization run.

    Parameters
    ----------
    document : `dict`
        document produced by utilities.prepare_document_global
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    fig, ax = share_fig_ax(fig, ax)
    _plot_attribute_global(document['rrmswfe_iter'], ax=ax)
    ax.set(ylabel=r'Residual RMS WFE [$\lambda$]')
    return fig, ax


def plot_mtf_focus_grid(data, frequencies, focus, fig=None, ax=None):
    """Plot a 2D view of MTF through frequency and focus.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D ndarray with dimensions of frequency,focus
    frequencies : iterable
        fequencies the data is given at; x axes, cy/mm
    focus : iterable
        focus points the data is given at; y axes, microns
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    xlims = (frequencies[0], frequencies[-1])
    ylims = (focus[0], focus[-1])
    fig, ax = share_fig_ax(fig, ax)
    ax.imshow(data,
              extent=[*xlims, *ylims],
              origin='lower',
              aspect='auto',
              interpolation='lanczos')
    ax.set(xlim=xlims, xlabel='Spatial Frequency $\nu$ [cy/mm]',
           ylim=ylims, ylabel='Focus [$\mu$ m]')

    return fig, ax


def plot_2d_optframe(data, max_freq=None, focus_range=None, focus_unit=r'$\mu m$', fig=None, ax=None):
    """Plot an image view of the data.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D data with first dimension spatial frequency, second dimension focus
    max_freq : `float`, optional
        maximum spatial frequency to plot, x axis range will be [0,max_freq]
    focus_range : `float`, optional
        maximum defocus value to plot, y axis range will be [-focus_range,focus_range]
    focus_unit : str, optional
        unit of focus, e.g. um or waves
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    fig, ax = share_fig_ax(fig, ax)

    im = ax.imshow(data,
                   extent=[0, max_freq, -focus_range, focus_range],
                   aspect='auto',
                   origin='lower',
                   interpolation=None)

    fig.colorbar(im)

    if max_freq is not None and focus_range is not None:
        ax.set(xlim=(0, max_freq), xlabel='Spatial Frequency [cy/mm]',
               ylim=(-focus_range, focus_range), ylabel=f'Focus [{focus_unit}]')

    return fig, ax


def linear_kde(data, xlim, num_pts=100, shade=True, bw_method=None, gridlines_below=True, fig=None, ax=None):
    """Create a Kernel Density Estimation based 'histogram' on a linear x axis.

    Parameters
    ----------
    data: `numpy.ndarray`
        data to plot
    xlim: iterable of length 2
        lower and upper x limits to plot
    num_pts: `int`, optional
        number of points to sample along x axis
    shade: `bool`, optional
        whether to shade the area under the curve
    bw_method: `str` or `float`, optional
        passed to `scipy.stats.gaussian_kde` to set the bandwidth during estimation
    gridlines_below: `bool`
        whether to set axis gridlines to be below the graphics
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    kde = gaussian_kde(data, bw_method)
    xpts = np.linspace(*xlim, num_pts)  # in transformed space
    ypts = kde(xpts)
    ypts = ypts / ypts.sum() * 100

    fig, ax = share_fig_ax(fig, ax)
    if shade is True:
        z = np.zeros(xpts.shape)
        ax.fill_between(xpts, ypts, z)
    ax.plot(xpts, ypts)
    ax.set(xlim=xlim, xlabel=r'Residual RMS WFE [$\lambda$]',
           ylim=(0, None), ylabel='Probability [%]', axisbelow=gridlines_below)
    return fig, ax


def log_kde(data, xlim, num_pts=100, shade=True, bw_method=None, gridlines_below=True, fig=None, ax=None):
    """Create a Kernel Density Estimation based 'histogram' on a logarithmic x axis.

    Parameters
    ----------
    data: `numpy.ndarray`
        data to plot
    xlim: iterable of length 2
        lower and upper x limits to plot
    num_pts: `int`, optional
        number of points to sample along x axis
    shade: `bool`, optional
        whether to shade the area under the curve
    bw_method: `str` or `float`, optional
        passed to `scipy.stats.gaussian_kde` to set the bandwidth during estimation
    gridlines_below: `bool`
        whether to set axis gridlines to be below the graphics
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    d = np.log10(data)
    kde = gaussian_kde(d, bw_method)
    xpts = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), num_pts)  # in transformed space
    data = kde(xpts)
    real_xpts = 10 ** xpts
    data = data / data.sum() * 100

    fig, ax = share_fig_ax(fig, ax)
    if shade is True:
        z = np.zeros(real_xpts.shape)
        ax.fill_between(real_xpts, data, z)
    ax.plot(real_xpts, data)
    ax.set(xlim=xlim, xlabel=r'Residual RMS WFE [$\lambda$]', xscale='log',
           ylim=(0, None), ylabel='Probability [%]', axisbelow=gridlines_below)
    return fig, ax


def plot_rmswfe_rrmswfe_scatter(db, ylim=(None, None), fig=None, ax=None):
    """Make a scatter plot of residual RMS WFE vs RMS WFE from a database.

    This plot can be used as a simplified means of understanding the capture
    range of MTF-based wavefront sensing.

    Parameters
    ----------
    db : `iris.data.Database`
        database of simulation results
    ylim : `iterable`, optional
        lower, upper y axis limits
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    fig, ax = share_fig_ax(fig, ax)
    ax.scatter(db.df.truth_rmswfe, db.df.rrmswfe_final)
    ax.set(xlabel=r'RMS WFE [$\lambda$]',
           ylim=ylim, ylabel=r'Residual RMS WFE [$\lambda$]', yscale='log')
    return fig, ax


def plot_final_cost_rrmswfe_scatter(db, ylim=(None, None), fig=None, ax=None):
    """Plot final cost residual RMS WFE vs final cost function value.

    This plot can be used as a simplified means of understanding a uniqueness
    problem in MTF-based wavefront sensing.

    Parameters
    ----------
    db : `iris.data.Database`
        database of simulation results
    ylim : iterable, optional
        Description
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    fig, ax = share_fig_ax(fig, ax)
    ax.scatter(db.df.cost_final, db.df.rrmswfe_final)
    ax.set(xlabel='Cost Function Value',
           ylim=ylim, ylabel=r'Residual RMS WFE [$\lambda$]', yscale='log')
    return fig, ax


def plot_axial_df_2d(df, titles=('Tangential', 'Sagittal'), fig=None, axs=None):

    axial_mtf_data = df[df.Field == 0]
    focuspos = np.unique(axial_mtf_data.Focus.as_matrix())
    ax_t = []
    ax_s = []
    for pos in focuspos:
        fd = axial_mtf_data[axial_mtf_data.Focus == pos]
        ax_t.append(fd[fd.Azimuth == 'Tan']['MTF'].as_matrix())
        ax_s.append(fd[fd.Azimuth == 'Sag']['MTF'].as_matrix())

    ax_t = np.asarray(ax_t)
    ax_s = np.asarray(ax_s)

    extx, exty = [df.Freq.min(), df.Freq.max()], [focuspos[0], focuspos[-1]]
    fig, axs = share_fig_ax(fig, axs, numax=2)
    for data, ax, title in zip([ax_t, ax_s], axs, titles):
        im = ax.imshow(data,
                  extent=[*extx, *exty],
                  origin='lower',
                  aspect='auto',
                  cmap='inferno',
                  clim=(0, 1),
                  interpolation='lanczos')
        ax.set(xlim=extx, xlabel='Spatial Frequency [cy/mm]', ylim=exty, ylabel=r'Focus [$\mu m$]', title=title)

    fig.tight_layout()
    fig.colorbar(im, ax=axs, label='MTF [Rel. 1.0]')
    return fig, axs
