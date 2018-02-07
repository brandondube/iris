"""Scrub through optimiation results."""
import sys
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Qt5agg')

from PyQt5.QtCore import Qt  # noqa
from PyQt5.QtWidgets import (  # noqa
    QApplication,
    QWidget,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QLabel,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.ticker import ScalarFormatter  # noqa
from matplotlib import pyplot as plt  # noqa

from prysm.thinlens import defocus_to_image_displacement  # noqa
from prysm.macros import thrufocus_mtf_from_wavefront_array  # noqa

from iris.core import config_codex_params_to_pupil  # noqa

root = Path(__file__).parent / '..' / '..' / '..' / 'simulations'

truth_data = root / 'truth_df.csv'
truth_wvfront = root / 'truth_wavefront.pkl'
sim_cfg = root / 'config.pkl'
sim_result = root / 'optimization_result.pkl'


# load truth data
df = pd.read_csv(truth_data)

# load truth wavefront
with open(truth_wvfront, 'rb') as fid:
    true_wvfront = pickle.load(fid)

# load configuration
with open(sim_cfg, 'rb') as fid:
    cfg = pickle.load(fid)

# load optimization history
with open(sim_result, 'rb') as fid:
    opt_res = pickle.load(fid)

# compute some metadata
epd = cfg.efl / cfg.fno
nit = len(opt_res['result_iter'])
focus_um = defocus_to_image_displacement(cfg.focus_range_waves, cfg.fno, cfg.wvl, cfg.focus_zernike, cfg.focus_normed)

CACHE = defaultdict(dict)
GAMMA = 1.5


def df_to_mtf_array(df, azimuth='Tan'):
    """Convert a dataframe to a 2D array of MTF data with axes of (freq,focus).

    Parameters
    ----------
    df : `pandas.DataFrame`
        a dataframe with a column for azimuth, focus, frequency, and MTF
    azimuth : `str`, optional, {'Tan', 'Sag'}
        which azimuth to grab

    Returns
    -------
    `numpy.ndarray`
        2d ndarray

    """
    return df[df.Azimuth == azimuth].as_matrix(columns=['MTF']).reshape(cfg.focus_planes, len(cfg.freqs))


OPTDATAKEY = 'mtfarr'
WVFRONTKEY = 'wvfront'


def populate_cache(iteration):
    """Populate the cache with MTF data for the given optimizer iteration.

    Parameters
    ----------
    iteration : `int`
        iteration to cache data for

    """
    # create the focused pupil and data array
    coefs, codex = opt_res['result_iter'][iteration], opt_res['codex']
    focus_pupil = config_codex_params_to_pupil(cfg, codex, coefs)
    data_t, data_s = thrufocus_mtf_from_wavefront_array(focus_pupil, cfg)

    # store them in the cache
    CACHE[iteration][WVFRONTKEY] = focus_pupil.phase
    CACHE[iteration][OPTDATAKEY] = data_t
    return data_t


class App(QMainWindow):
    """Optimizaton Result Scrubber app.

    Allows user to scrub through optimization results with a slider and view a large volume of data
    for each tick of the optimizer.

    Attributes
    ----------
    height : `int`
        window height, px
    left : `int`
        window left position, px
    title : `str`
        window title
    top : `int`
        window top position, px
    width : `int`
        window width, px

    """

    def __init__(self):
        """Create a new Optimization Result Scrubber."""
        super(App, self).__init__()

        # set size and title
        self.left = 20
        self.top = 60
        self.width = 1600
        self.height = 800
        self.title = 'Optimization Explorer'
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # make the main layout
        self.main = QWidget()
        self.setCentralWidget(self.main)
        self.main_layout = QVBoxLayout()
        self.plot_layout = QHBoxLayout()
        self.slider_layout = QHBoxLayout()
        self.main.setLayout(self.main_layout)
        self.main_layout.addLayout(self.plot_layout)
        self.main_layout.addLayout(self.slider_layout)

        self.init_optdata_plot()
        self.init_wavefront_plot()
        self.init_costfunction_plot()
        self.init_slider()
        self.show()

    def init_optdata_plot(self):
        """Initialize the optimization data plot."""
        self.opt_data_canvas = FigureCanvas(plt.figure())
        self.opt_fig = self.opt_data_canvas.figure
        self.truth_ax, self.citer_ax = self.opt_fig.subplots(nrows=2)

        # draw the truth image
        truth_dat = df_to_mtf_array(df, 'Tan')
        self.dat_ext = [0, cfg.freqs[-1], -focus_um, focus_um]

        truth_im = self.truth_ax.imshow(truth_dat,
                                        extent=self.dat_ext,
                                        aspect='auto',
                                        origin='lower',
                                        cmap='inferno',
                                        norm=mpl.colors.PowerNorm(1 / GAMMA),
                                        interpolation='lanczos',
                                        vmin=0, vmax=1)
        self.truth_ax.set(ylabel=r'Focus [$\mu m$]', title='Truth')
        self.citer_ax.set(xlabel='Spatial Frequency [cy/mm]', title='Current Iteration')

        # draw the first iteration image
        imdat = populate_cache(0)
        self.iter_dat_im = self.citer_ax.imshow(imdat,
                                                extent=self.dat_ext,
                                                aspect='auto',
                                                origin='lower',
                                                cmap='inferno',
                                                interpolation='lanczos',
                                                norm=mpl.colors.PowerNorm(1 / GAMMA),
                                                vmin=0, vmax=1)

        # draw the colorbar
        self.opt_fig.tight_layout()
        self.opt_fig.subplots_adjust(right=0.8)
        cbax = self.opt_fig.add_axes([.8125, 0.05, .05, .9])
        self.cb = self.opt_fig.colorbar(truth_im, cax=cbax)
        if GAMMA != 1.0:
            raised_txt = f', raised to 1/{GAMMA} power'
        else:
            raised_txt = ''
        cbax.set(ylabel=f'MTF [Rel. 1.0{raised_txt}]')

        self.plot_layout.addWidget(self.opt_data_canvas)

    def init_wavefront_plot(self):
        """Initialize the wavefront plots."""
        self.pup_ext = [-epd, epd, -epd, epd]
        self.wvfront_canvas = FigureCanvas(plt.figure())
        self.wave_fig = self.wvfront_canvas.figure
        self.true_wv_ax, self.citer_wv_ax = self.wave_fig.subplots(nrows=2)

        # draw the truth wavefront
        nani = true_wvfront.phase[np.isfinite(true_wvfront.phase)]
        mx, mn = nani.max(), nani.min()
        truth_im = self.true_wv_ax.imshow(true_wvfront.phase,
                                          extent=self.pup_ext,
                                          origin='lower',
                                          cmap='RdYlBu',
                                          interpolation='lanczos',
                                          vmin=mn, vmax=mx)
        self.true_wv_ax.set(ylabel=r'Pupil $\eta$ [mm]', title='Truth')
        self.citer_wv_ax.set(xlabel=r'Pupil $\xi$ [mm]', title='Current Iteration')

        # draw the first iteration image
        imdat = CACHE[0][WVFRONTKEY]
        self.iter_wv_im = self.citer_wv_ax.imshow(imdat,
                                                  extent=self.pup_ext,
                                                  origin='lower',
                                                  cmap='RdYlBu',
                                                  interpolation='lanczos',
                                                  vmin=mn, vmax=mx)

        # make room and draw the colorbar
        self.wave_fig.tight_layout()
        self.wave_fig.subplots_adjust(left=0.25)
        cbax = self.wave_fig.add_axes([.175, 0.05, .05, .9])
        self.cb = self.wave_fig.colorbar(truth_im, cax=cbax)
        cbax.set(ylabel=r'OPD [$\lambda$]')
        cbax.yaxis.set_ticks_position('left')
        cbax.yaxis.set_label_position('left')

        self.plot_layout.addWidget(self.wvfront_canvas)

    def init_costfunction_plot(self):
        """Initialize the plot of the cost function and residual RMS WFE vs iteration."""
        with mpl.style.context('ggplot'):
            self.cost_canvas = FigureCanvas(plt.figure())
            self.cost_fig = self.cost_canvas.figure
            self.cost_axis, self.rmswfe_axis = self.cost_fig.subplots(nrows=2, sharex=True)
            iters = range(nit)

            self.cost_axis.plot(iters, opt_res['cost_iter'], lw=3)
            self.cost_highlight, = self.cost_axis.plot(0, opt_res['cost_iter'][0], '.', ms=15)
            self.cost_axis.set(ylabel='Cost Function Value [a.u.]')

            self.rmswfe_axis.plot(iters, opt_res['rmswfe_iter'], lw=3)
            self.rmswfe_highlight, = self.rmswfe_axis.plot(0, opt_res['rmswfe_iter'][0], '.', ms=15)
            self.rmswfe_axis.set(xlabel='Iteration', ylabel=r'Residual RMS WFE [$\lambda$]')
            self.rmswfe_axis.xaxis.set_major_formatter(ScalarFormatter())
            self.cost_fig.tight_layout()

        self.plot_layout.addWidget(self.cost_canvas)

    def init_slider(self):
        """Create and initialize the slider."""
        self.iteration_slider = QSlider(Qt.Horizontal)
        self.iteration_slider.setFocusPolicy(Qt.StrongFocus)
        self.iteration_slider.setTickInterval(1)
        self.iteration_slider.setSingleStep(1)
        self.iteration_slider.setMinimum(0)
        self.iteration_slider.setMaximum(nit - 1)
        self.iteration_slider.setTickPosition(QSlider.TicksBelow)
        self.iteration_slider.valueChanged.connect(self.update)

        self.slider_value_label = QLabel('00')
        self.slider_layout.addWidget(self.slider_value_label)
        self.slider_layout.addWidget(self.iteration_slider)

    def update(self):
        """Update the UI from the current iteration number."""
        self.iteration = self.iteration_slider.value()

        # check if the cache image data has been made
        if self.iteration not in CACHE:
            populate_cache(self.iteration)

        self.update_slidertext()
        self.update_optdata_plot()
        self.update_wavefront_plot()
        self.update_cost_plot()

        arr_true = df_to_mtf_array(df, 'Tan')
        arr_00 = CACHE[self.iteration]['mtfarr']
        costfcn = ((arr_true - arr_00) ** 2).sum()
        print(costfcn)

    def update_slidertext(self):
        """Update the slider text."""
        self.slider_value_label.setText(f'{self.iteration:02d}')

    def update_optdata_plot(self):
        """Update the optimization data plot."""
        self.iter_dat_im.set_data(CACHE[self.iteration][OPTDATAKEY])
        self.opt_fig.canvas.draw_idle()

    def update_wavefront_plot(self):
        """Update the wavefront data plot."""
        self.iter_wv_im.set_data(CACHE[self.iteration][WVFRONTKEY])
        self.wave_fig.canvas.draw_idle()

    def update_cost_plot(self):
        """Update the cost function plot."""
        self.cost_highlight.set_xdata(self.iteration)
        self.cost_highlight.set_ydata(opt_res['cost_iter'][self.iteration])
        self.rmswfe_highlight.set_xdata(self.iteration)
        self.rmswfe_highlight.set_ydata(opt_res['rmswfe_iter'][self.iteration])
        self.cost_fig.canvas.draw_idle()


if __name__ == '__main__':
    qapp = QApplication(sys.argv)
    app = App()
    sys.exit(qapp.exec_())
