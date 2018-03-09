"""Tools for plotting the results of wavefront sensing."""
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

from prysm.util import share_fig_ax


def single_solve_triple(result_document, log=False, fig=None, axs=None):
    """Plot a quadchart for a single solve, giving full diagnostic information into the result.

    Parameters
    ----------
    result_document : `dict`
        `dict` optimization result including initial values
    log : bool, optional
        whether to use a logarithmic scale
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    """
    if log is False:
        scale = 'linear'
    else:
        scale = 'log'

    # pull the relevant data
    rd = result_document
    params, cost, rmswfe = rd['result_iter'], rd['cost_iter'], rd['rmswfe_iter']
    iters = list(range(len(params)))
    params = np.asarray(params)
    p_shape = params.shape

    # prepare the plot
    fig, axs = plt.subplots(ncols=3, sharex=True, figsize=(12, 4))

    # get the parameter names
    names = list(rd['codex'].values())
    truths = rd['truth_params']
    for i, name, truth in zip(range(p_shape[1]), names, truths):
        line, = axs[0].plot(iters, params[:, i], label=f'{name} : {truth}')
        axs[0].scatter(iters[-1], truth, c=line.get_color(), linewidths=3)

    axs[0].legend()
    axs[0].set(xlabel='Iteration [-]',
               ylabel=r'Zernike Weight [$\lambda$ RMS]',
               title='Parameters')

    axs[1].plot(iters, cost)
    axs[1].set(xlabel='Iteration [-]',
               ylabel='Value [-]',
               yscale=scale,
               title='Cost Function',)

    axs[2].plot(iters, rmswfe)
    axs[2].set(xlabel='Iteration [-]',
               ylabel=r'Residual RMS WFE [$\lambda$]',
               yscale=scale,
               title='RMS WFE')

    fig.tight_layout()
    return fig, axs


def single_solve_paper(result_document, fig=None, axs=None):
    """Plot the convergence of parameters on a symmetric log scale.

    Parameters
    ----------
    result_document : `dict`
        `dict` optimization result including initial values
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot


    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    """
    rd = result_document
    truth = rd['truth_zernike']
    truths = np.asarray(list(truth.values()), dtype=np.float64)
    params, rmswfe = rd['x_iter'], rd['rmswfe_iter']
    iters = list(range(len(params)))
    params = np.asarray(params)
    p_shape = params.shape

    # prepare the plot
    fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(8, 3.75))

    # get the parameter names
    names = list(result_document['retrieved_zernike'].keys())
    truths = list(result_document['truth_zernike'].values())
    for i, name, truth in zip(range(p_shape[1]), names, truths):
        line, = axs[0].plot(iters, params[:, i], label=f'{name} : {truth}')
        axs[0].scatter(iters[-1], truth, c=line.get_color(), linewidths=2.5)

    axs[0].legend()
    axs[0].set(xlabel='Optimizer Iteration',
               ylabel=r'Zernike Weight [$\lambda$ 0-P]')

    axs[1].plot(iters, rmswfe, c='0.25')
    axs[1].set(xlabel='Optimizer Iteration',
               ylabel=r'Residual RMS WFE [$\lambda$]',
               yscale='log')

    fig.tight_layout()
    return fig, axs


def plot_costfcn_by_iter(results, fig=None, ax=None):
    """Plot the cost function by iteration.

    Parameters
    ----------
    results : `dict`
        results dictionary, with fun_iter key
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    """
    cost = results['fun_iter']
    x = range(len(cost))
    fig, ax = share_fig_ax(fig, ax)
    ax.plot(x, cost, lw=3)

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
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
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
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
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
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`:
        Axis containing the plot

    """
    d = np.log10(data)
    kde = gaussian_kde(d, bw_method)
    xpts = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), num_pts)  # in transformed space
    data = kde(xpts)
    real_xpts = 10 ** xpts

    fig, ax = share_fig_ax(fig, ax)
    if shade is True:
        z = np.zeros(real_xpts.shape)
        ax.fill_between(real_xpts, data, z)
    ax.plot(real_xpts, data)
    ax.set(xlim=xlim, xlabel=r'Residual RMS WFE [$\lambda$]', xscale='log',
           ylim=(0, None), ylabel='Probability Density', axisbelow=gridlines_below)
    return fig, ax
