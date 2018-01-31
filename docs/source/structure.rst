#########
Structure
#########

iris has a somewhat novel architecture designed around being high performance and easily modified to suit the needs of a problem.  The core defines:

- a cost function

- a cost function modifier [#]_

- realization of MTF at various focus planes

- notionally in focus wavefront realization from modal Fringe Zernike parameters

- "broadcasting" of this wavefront to all focus planes

- user-provided simulation configuration

- - EFL

- - F/#

- - :math:`\lambda`

- - number of focus planes

- - focus range

- - specific spatial frequencies to evaluate

- - pupil plane samples

.. [#] This works very similarly to middleware in web server programming.


Wavefront Sensing Routine
=========================

Wavefront sensing has three general phases:

#.  Preparation / setup
#.  Optimization
#.  Cleanup and post processing

Each is described below.

Preparation
-----------

* preprocess truth data (format conversion, denoise/smoothing, etc).
* Create a set of defocus pupils.
* create a list and append the initial guess.
* create a "decoder ring" of parameter vector positions and zernike names (e.g. 3 -> Z9).
* spin up a pool of worker processes to realize focus planes.
* assign the following global variables

    * truth data
    * configuration data (EFL, wavelength, F/#, etc)
    * defocus pupils
    * diffraction limited MTF

* start timer.

Optimization
------------

* forcefully redirect stdout to capture disp statements from the optimizer.
* generate the "notionally in focus" pupil for the given parameter vector.
* merge it with each defocus PSF and propogate each to its own MTF.
* compute the cost function, where :math:`C_i` denotes "initial cost."

.. math::

    C_i = \sum_{\text{focus}} \, \sum_{\nu=10}^{\nu_\text{c}} \left(\text{D} - \text{M}\right)^2

* apply the cost function modifier, where :math:`C_f` denotes "final cost." [#]_

.. math::
    C_f = \frac{d\nu}{N_{\text{focus}}}C_i

.. [#] :math:`\sum x dx` is a popular numerical integration technique; this modifier changes the meaning of the cost function to be integration over the spatial frequency dimension and normalizes by the number of focus planes.

* after each iteration, append the parameter vector chosen at the end of line search to the list initialized in preparation.

Cleanup
-------

* stop timer.
* parse L-BFGS-B disp statements for cost function by iteration.
* compute residual RMS WFE by iteration.
* store all results in a dictionary.

Performance Optimizations
=========================

iris has some ugly coding practices that are used to facilitiate high performance.  In sum, they allow approximately **15x** performance gain over not using them and their practice is warranted for the desired level of speed for this library.  They are listed below.  Some may be considered more red flags, but it should be noted that there is nothing inherently wrong in the use of these techniques; they are only advised against as a pragma in programming classes and circles.

Global Variables
----------------

Numpy is not only an ndarray library, but also a runtime engine that handles array operations in C or lower level instruction sets.  It is not obvious to the user when numpy will choose to copy or reference an array; often, when an array is not an attribute of a class instance and is used as an argument in a class instance method, numpy will trigger a copy on access of the array.  For small arrays or in non parllel environments, this may have minimal impact.  When 8 high memory bandwidth processes are competing for page times, this can create a significant performance impact.  For example, performing a :math:`\sin` operation with vectorized instructions on 64-bit floats, as numpy linked to MKL or OpenBLAS will do, occurs on up to 16 items at a time per core.  This is equal to :math:`16 \cdot \tfrac{64}{8}` Bytes per clock per core.  At 3.5Ghz, this would be equal to 448GB/s of number crunching.  The highest speed memory available today can produce about 25GB/s, so this operation is severely memory limited.  Other operations (python overhead, FFTs, etc) are slower and indeed not memory bandwidth limited, so there is a net gain in multi-core operation.  The key is to avoid unnecessary movement of large arrays of data.

A solution is to not pass these arrays as function arguments, but to make them global variables which are referenced inside the body of the program.  This prevents the triggering of any copy numpy semantics, and boosts performance.  It also avoids the the use :func:`functools.partial` to create single-argument (optimization parameter vector) functions of multiple variables (optimization parameter vector, configuration dictionary, truth data, etc).

Implementing usage of limited global variables increased performance by approximately **12%**.


Shared Global State
-------------------

iris is fully multithreaded.  In python, this means the use of multiple processes which do not have shared memory.  As a result, the global variables shared by processes are not so global after all and changes in one process will not be reflected in others.  Iris strictly uses read-only practices with globals, so this becomes a nonissue.  This is a byproduct of the use of global variables more than an optimization.

Use of multiple worker processes to realize focal planes in parallel increased performance by approximately **550%** on a 4-core, 8-thread machine.  About **200%** of this gain was made through reduced memory thrashing enabled by the global variable usage above; those gains do not appear in single-core operation.

Precomputation of defocus
-------------------------

iris will compute each pupil associated with the defocus for a given plane ahead of time, and store the set of them in global memory.  This is duplicated across all workers and read.  These pupils are added to the notionally in focus pupil to perform each propagation.  In this way, the defocus terms are not unnecessarily recomputed in each optimization iteration.

This increased performance by approximately **7%**.

Lack of Partial callables
-------------------------

A :func:`~functools.partial` is a callable that wraps another function with some fixed arguments.  Callables have additional overhead when the are called; eliminating their use improved performance by approximately **2%**.  Bigger gains were made to the simplicity and legibility of the code.


Choice of Q
-----------

Q, the oversampling factor, is chosen to be 2 explicitly in these computations as higher is not needed for a faithful MTF computation within the bandwidth imposed by diffraction.  This allows the fastest possible propagation for a given array size.

Moving from Q of 3 to Q of 2 increased performance by approximately **125%**.

Smart Interpolation
-------------------

prysm previously only allowed full 2D interpolation of MTF data.  The generation of the interpolation function not along the cardinal axes was wasteful.  A feature was added that allows creation of interpolation functions strictly along the x=0 and y=0 axes if only tangential and sagittal MTF is needed at exact (interpolated) frequencies.  This boosted performance by approximately **30%**.

Numba
-----

`Numba <http://numba.pydata.org/>`_ is a JIT compiler for python.  Its usage in prysm allows the merging of math kernels for Zernike functions, increasing the amount of work done to memory as it is recieved.  Stated differently, numba increases CPU demand by more densely packing computations, bringing it more in-line with memory throughput demand.
