''' Matplotlib figures that represent a panel used to monitor the convergence
    of an optimization routine.
'''

from matplotlib import pyplot as plt

class ConvergencePanel(object):
    ''' A mutatable panel used to plot the convergence of a nonlinear
        optimization routine.
    '''
    def __init__(self, cost_function=None, plt_labels=None, plt_data=None, plt_lines=None, iters=[]):
        # observers are called on data changes (this automatically triggers the
        # parameter and cost function plot(s) on a data change )
        self.observers = []

        self.cost_function = cost_function
        self.plt_labels = plt_labels
        self.plt_data = plt_data
        self.plt_lines = plt_lines
        self.iters = iters

        self.iters_current = 0
        self.ymax_last = 0
        self._current_data = None

        if plt_labels is None:
            self.plt_add_legend = False
        else:
            self.plt_add_legend = True
        
        self.fig, self.ax = plt.subplots()
        self.param_ax = self.ax
        #self.param_ax = self.axs[0]
        plt.ion()
        self.param_ax.set(xlabel='Generation',
                          ylabel='Parameter Value')

    @property
    def current_data(self):
        return self._current_data

    @current_data.setter
    def current_data(self, data):
        self._current_data = data
        self.update_param_plot(data)
    
    def add_labels(self, labels):
        self.plt_labels = labels
        self.plt_add_legend = True
    
    def initiate_lines(self, data):
        if self.plt_data is None:
            # make arrays for the parameter values over generations
            self.plt_data = [list() for x in range(len(data))]
            self.plt_lines = []

            # adapt line width to density of plot
            length = len(data)
            line_width = 3
            if len(data) > 5:
                line_width = 2
            if len(data) > 10:
                line_width = 10
            
            # create line objects
            for idx in range(len(data)):
                line, = self.param_ax.plot([], [], lw=line_width)
                self.plt_lines.append(line)

    def update_param_plot(self, new_data):
        # initiate the lines if they do not exist
        if self.plt_data is None:
            self.initiate_lines(new_data)
        
        # increment the iterations
        self.iters.append(self.iters_current)
        self.iters_current += 1

        # append the current data
        for idx, val in enumerate(new_data):
            self.plt_data[idx].append(val)
        
        for line, dat in zip(self.plt_lines, self.plt_data):
                line.set_xdata(self.iters)
                line.set_ydata(dat)
        
        # handle legend
        # TODO: refactor
        if self.plt_add_legend and self.plt_data is not None:
            ncols = 1
            if len(self.plt_labels) > 6:
                ncols = 2
            elif len(self.plt_labels > 9):
                ncols = 3
            elif len(self.plt_labels > 12):
                ncols = 4
            self.param_ax.legend(self.plt_labels, ncol=ncols)
            self.plt_add_legend = False
        
        # recompute the ax.dataLim
        self.param_ax.relim()
        # update ax.viewLim using the new dataLim
        self.param_ax.autoscale_view()

        # avoid overflow from axis limits
        maxs = [max(vector) for vector in self.plt_data]
        ymax = max(maxs)
        if len(str(ymax)) > len(str(self.ymax_last)):
            self.fig.tight_layout()
        ymax_last = ymax
        plt.pause(0.0001)