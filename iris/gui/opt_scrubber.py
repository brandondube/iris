"""Scrub through optimiation results."""
import sys
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Qt5agg')

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
from matplotlib import pyplot as plt  # noqa

from prysm.thinlens import defocus_to_image_displacement  # noqa
from prysm.macros import thrufocus_mtf_from_wavefront  # noqa

from iris.core import config_codex_params_to_pupil  # noqa

root = Path(__file__).parent / '..' / '..' / '..' / 'simulations'

truth_data = root / 'truth_df.csv'
sim_cfg = root / 'config.pkl'
sim_result = root / 'opt_result.pkl'

if True:
    # load truth data
    df = pd.read_csv(truth_data)

    # load configuration
    with open(sim_cfg, 'rb') as fid:
        cfg = pickle.load(fid)

    # load optimization history
    with open(sim_result, 'rb') as fid:
        opt_res = pickle.load(fid)

    # compute some metadata
    nit = len(opt_res['result_iter'])
    focus_um = defocus_to_image_displacement(cfg['focus_range_waves'], cfg['fno'], cfg['wavelength'])

CACHE = defaultdict(dict)


def df_to_mtf_array(df, azimuth='Tan'):
    """Convert a dataframe to a 2D array of MTF data with axes of (freq,focus).

    Parameters
    ----------
    df : `pandas.DataFrame`
        a dataframe with a column for azimuth, focus, frequency, and MTF
    azimuth : `str`, optional, {'Tan', 'Sag'}
        which azimuth to grab;

    Returns
    -------
    `numpy.ndarray`
        2d ndarray

    """
    return df[df.Azimuth == azimuth].as_matrix(columns=['MTF']).reshape(cfg['focus_planes'], len(cfg['freqs']))


def populate_cache(iteration):
    """Populate the cache with MTF data for the given optimizer iteration.

    Parameters
    ----------
    iteration : `int`
        iteration to cache data for

    """
    coefs, codex = opt_res['result_iter'][iteration], opt_res['codex']
    focus_pupil = config_codex_params_to_pupil(cfg, codex, coefs)
    data = thrufocus_mtf_from_wavefront(focus_pupil, cfg)
    arr = df_to_mtf_array(data, 'Tan')
    CACHE[iteration]['mtfarr'] = arr
    return arr


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
        # initialize the window
        super(App, self).__init__()

        # set size and title
        self.left = 20
        self.top = 60
        self.width = 640
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

        # create a figure and two axes used to draw the truth data and the current iteration data
        self.opt_data_canvas = FigureCanvas(plt.figure())
        self.opt_fig = self.opt_data_canvas.figure
        self.truth_ax, self.citer_ax = self.opt_fig.subplots(nrows=2)

        # draw the truth image
        truth_dat = df_to_mtf_array(df, 'Tan')
        self.dat_ext = [0, cfg['freqs'][-1], -focus_um, focus_um]

        truth_im = self.truth_ax.imshow(truth_dat,
                                        extent=self.dat_ext,
                                        aspect='auto',
                                        origin='lower',
                                        cmap='inferno',
                                        vmin=0, vmax=1)
        self.truth_ax.set(ylabel=r'Focus [$\mu m$]', title='Truth')
        self.citer_ax.set(xlabel='Spatial Frequency [cy/mm]', title='Current Iteration')

        # draw the first iteration image
        imdat = populate_cache(0)
        self.iter_im = self.citer_ax.imshow(imdat,
                                            extent=self.dat_ext,
                                            aspect='auto',
                                            origin='lower',
                                            cmap='inferno',
                                            vmin=0, vmax=1)

        # draw the colorbar
        self.opt_fig.tight_layout()
        self.opt_fig.subplots_adjust(right=0.8)
        cbax = self.opt_fig.add_axes([.85, 0.05, .05, .9])
        self.cb = self.opt_fig.colorbar(truth_im, cax=cbax)
        cbax.set(ylabel='MTF [Rel. 1.0]')

        # add a label for the slider
        self.slider_value_label = QLabel('00')

        # add a slider
        self.iteration_slider = QSlider(Qt.Horizontal)
        self.iteration_slider.setFocusPolicy(Qt.StrongFocus)
        self.iteration_slider.setTickInterval(1)
        self.iteration_slider.setSingleStep(1)
        self.iteration_slider.setMinimum(0)
        self.iteration_slider.setMaximum(nit - 1)
        self.iteration_slider.setTickPosition(QSlider.TicksBelow)
        self.iteration_slider.valueChanged.connect(self.update)

        # do the initial draw
        self.plot_layout.addWidget(self.opt_data_canvas)

        self.slider_layout.addWidget(self.slider_value_label)
        self.slider_layout.addWidget(self.iteration_slider)
        self.show()

    def update(self):
        """Update the UI from the current iteration number."""
        iteration = self.iteration_slider.value()

        # set the slider text
        self.slider_value_label.setText(f'{iteration:02d}')

        # check if the cache image data has been made
        if iteration in CACHE:
            imdat = CACHE[iteration]['mtfarr']
        else:
            imdat = populate_cache(iteration)

        self.iter_im.set_data(imdat)
        self.opt_fig.canvas.draw()


if __name__ == '__main__':
    qapp = QApplication(sys.argv)
    app = App()
    sys.exit(qapp.exec_())
