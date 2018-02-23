"""Tests scaling with multiple CPU cores to find the ideal number of threads."""
from iris.macros import run_simulation

if __name__ == '__main__':
    out = []
    for nthread in range(24):
        if nthread is 0:
            print('skipping 0')
            continue
        result = run_simulation(
            truth=(0, 0.2, -0.1, 0.05),
            solver='global',
            solver_opts={
                'parallel': True,
                'nthreads': nthread + 1})
        t = result['time']
        string = f'{nthread + 1} threads, {t:.3f}s runtime'
        out.append(string)

    print('\n'.join(out))
