''' Tools for plotting the results of phase retrieval
'''
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def single_solve_triple(result_document, type='axis_sph', log=False, fig=None, axs=None):
    ''' Plots a quadchart for a single solve, giving full diagnostic information
        into the result.

    '''
    if log is False:
        scale = 'linear'
    else:
        scale = 'log'

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
    ''' Plots the convergence of parameters on a symmetric log scale
    '''

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
