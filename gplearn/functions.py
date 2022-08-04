"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from joblib import wrap_non_picklable_objects

import talib as ta
import pandas as pd
from scipy import stats

__all__ = ['make_function']


class _Function(object):
    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, is_ts=False, params_need=None):
        self.function = function
        self.name = name
        self.arity = arity

        ##新增参数
        self.is_ts = is_ts
        self.d = 10
        self.params_need = params_need

        # print('init name : ', self.name)
        # print('init is_ts : ', self.is_ts)
        # print('init d : ', self.d)

    def __call__(self, *args):
        if not self.is_ts:
            # print('no ts name : ', self.name)
            return self.function(*args)

        else:
            # print('_Function call')
            # print('name : ', self.name)
            if self.d == 0:
                # print('self.d == 0:')
                raise AttributeError('Please reset attribute "d"')
            else:
                # print('self.d != 0')
                return self.function(*args, self.d)

    def set_d(self, d):
        self.d = d
        self.name += '_%d' % self.d
        # print('set_d d : ', self.d)
        # print('sed_d name : ', self.name)


def make_function(function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def _ts_delay(x1, d):
    return pd.Series(x1).shift(d).values


ts_delay1 = _Function(function=_ts_delay, name='ts_delay', arity=1, is_ts=True)


def _ts_delta(x1, d):
    return x1 - _ts_delay(x1, d)


ts_delta1 = _Function(function=_ts_delta, name='ts_delta', arity=1, is_ts=True)


def _ts_min(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).min()


ts_min1 = _Function(function=_ts_min, name='ts_min', arity=1, is_ts=True)


def _ts_max(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).max()


ts_max1 = _Function(function=_ts_max, name='ts_max', arity=1, is_ts=True)


def _ts_argmin(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(lambda x: x.argmin())


ts_argmin1 = _Function(function=_ts_argmin, name='ts_argmin', arity=1, is_ts=True)


def _ts_argmax(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(lambda x: x.argmax())


ts_argmax1 = _Function(function=_ts_argmax, name='ts_argmax', arity=1, is_ts=True)


def _ts_rank(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(
        lambda x: stats.percentileofscore(x, x[x.last_valid_index()]) / 100
    )


ts_rank1 = _Function(function=_ts_rank, name='ts_rank', arity=1, is_ts=True)


def _ts_sum(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).sum()


ts_sum1 = _Function(function=_ts_sum, name='ts_sum', arity=1, is_ts=True)


def _ts_stddev(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).std()


ts_stddev1 = _Function(function=_ts_stddev, name='ts_stddev', arity=1, is_ts=True)


def _ts_corr(x1, x2, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).corr(pd.Series(x2))


ts_corr2 = _Function(function=_ts_corr, name='ts_corr', arity=2, is_ts=True)


def _ts_mean_return(x1, d):
    return pd.Series(x1).pct_change().rolling(d, min_periods=int(d / 2)).mean()


ts_mean_return1 = _Function(function=_ts_mean_return, name='ts_mean_return',
                            arity=1, is_ts=True)

ts_dema1 = _Function(function=ta.DEMA, name='DEMA', arity=1, is_ts=True)

ts_kama1 = _Function(function=ta.KAMA, name='KAMA', arity=1, is_ts=True)

ts_ma1 = _Function(function=ta.MA, name='MA', arity=1, is_ts=True)

ts_midpoint1 = _Function(function=ta.MIDPOINT, name='MIDPOINT', arity=1, is_ts=True)

ts_beta2 = _Function(function=ta.BETA, name='BETA', arity=2, is_ts=True)

ts_lr_angle1 = _Function(function=ta.LINEARREG_ANGLE, name='LR_ANGLE',
                         arity=1, is_ts=True)

ts_lr_intercept1: _Function = _Function(function=ta.LINEARREG_INTERCEPT,
                                        name='LR_INTERCEPT', arity=1, is_ts=True)

ts_lr_slope1 = _Function(function=ta.LINEARREG_SLOPE, name='LR_SLOPE',
                         arity=1, is_ts=True)

# ts_ht1 = _Function(function=ta.HT_DCPHASE, name='HT', arity=1, is_ts=True)

add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

# _function_map = {'add': add2,
#                 'sub': sub2,
#                 'mul': mul2,
#                 'div': div2,
#                 'sqrt': sqrt1,
#                 'log': log1,
#                 'abs': abs1,
#                 'neg': neg1,
#                 'inv': inv1,
#                 'max': max2,
#                 'min': min2,
#                 'sin': sin1,
#                 'cos': cos1,
#                 'tan': tan1}

_function_map = {
    'add': add2,
    'sub': sub2,
    'mul': mul2,
    'div': div2,
    'sqrt': sqrt1,
    'log': log1,
    'abs': abs1,
    'neg': neg1,
    'inv': inv1,
    'max': max2,
    'min': min2,
    'sin': sin1,
    'cos': cos1,
    'tan': tan1,
    'ts_delay': ts_delay1,
    'ts_delta': ts_delta1,
    'ts_min': ts_min1,
    'ts_max': ts_max1,
    'ts_argmin': ts_argmin1,
    'ts_argmax': ts_argmax1,
    'ts_rank': ts_rank1,
    'ts_stddev': ts_stddev1,
    'ts_corr': ts_corr2,
    'ts_mean_return': ts_mean_return1,

    'DEMA': ts_dema1,
    'KAMA': ts_kama1,
    'MA': ts_ma1,
    'MIDPOINT': ts_midpoint1,
    'BETA': ts_beta2,
    'LR_ANGLE': ts_lr_angle1,
    'LR_INTERCEPT': ts_lr_intercept1,
    'LR_SLOPE': ts_lr_slope1,
    # 'HT': ts_ht1
}
