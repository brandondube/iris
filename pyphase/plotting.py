''' Tools for plotting the results of phase retrieval
'''
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.style.use('ggplot')
mpl.rcParams.update({'lines.linewidth': 3})

def single_solve_quad(result_document, type='axis_sph', fig=None, axs=None):
    ''' Plots a quadchart for a single solve, giving full diagnostic information
        into the result.

    '''
    # pull the relevant data
    rd = result_document
    params, cost, rmswfe = rd['x_iter'], rd['fun_iter'], rd['rmswfe_iter']
    iters = list(range(len(params)))
    params = np.asarray(params)
    p_shape = params.shape

    # prepare the plot
    fig, axs = plt.subplots(ncols=3, sharex=True, figsize=(12, 4))

    # get the parameter names
    names = list(result_document['retrieved_zernike'].keys())
    truths = list(result_document['truth_zernike'].values())
    for i, name, truth in zip(range(p_shape[1]), names, truths):
        line, = axs[0].plot(iters, params[:, i], label=f'{name} : {truth}')
        axs[0].scatter(iters[-1], truth, c=line.get_color(), linewidths=3)

    axs[0].legend()
    axs[0].set(xlabel='Iteration [-]',
               ylabel=r'Zernike Weight [$\lambda$ 0P]',
               title='Parameters')

    axs[1].plot(iters, cost)
    axs[1].set(xlabel='Iteration [-]',
               ylabel='Value [-]',
               title='Cost Function')

    axs[2].plot(iters, rmswfe)
    axs[2].set(xlabel='Iteration [-]',
               ylabel=r'Residual RMS WFE [$\lambda$]',
               title='RMS WFE')

    fig.tight_layout()
    return fig, axs
