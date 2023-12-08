import itertools
import functools
import numpy as np
from autograd.misc import flatten

def unflatten_optimizer(optimizer):
    @functools.wraps(optimizer)
    def optimize(grad, x, *args, **kwargs):
        _x, x_unflatten = flatten(x)
        _grad = lambda x, i: flatten(grad(x_unflatten(x), i))[0]
        opt = optimizer(_grad, _x, *args, **kwargs)
        val = None
        while 1:
            val = opt.send(val)
            val = yield (x_unflatten(val[0]), val[1], x_unflatten(val[2]))
    return optimize

@unflatten_optimizer
def universal(grad, x, step_size_gen, b1, b2, eps = 10**-8):
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for n in itertools.count():
        g = grad(x, n)
        yield x, n, g
        m = m * b1 + g
        v = v * b2 + g**2
        x = x - step_size_gen.send(n) * m / np.sqrt(v + eps)

@unflatten_optimizer
def adagrad(grad, x, step_size=0.001, eps = 10**-8):
    def gen():
        while 1:
            yield step_size
    gg = gen()
    gg.send(None)
    return universal(grad, x, gg, 0, 1, eps)

@unflatten_optimizer
def adam(grad, x, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    def gen():
        value = None
        while 1:
            n = yield value
            value = step_size
            value *= (1-b1)
            value /= np.sqrt(1-b2)
            value /= (1-b1**(n+1)) # m correction
            value *= np.sqrt(1-b2**(n+1)) # v correction
    gg = gen()
    gg.send(None)
    return universal(grad, x, gg, b1, b2, eps)

@unflatten_optimizer
def adam_no_m(grad, x, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    def gen():
        value = None
        while 1:
            n = yield value
            value = step_size
            value *= (1-b1)
            value /= np.sqrt(1-b2)
            # value /= (1-b1**(n+1)) # m correction
            value *= np.sqrt(1-b2**(n+1)) # v correction
    gg = gen()
    gg.send(None)
    return universal(grad, x, gg, b1, b2, eps)

@unflatten_optimizer
def adam_no_v(grad, x, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    def gen():
        value = None
        while 1:
            n = yield value
            value = step_size
            value *= (1-b1)
            value /= np.sqrt(1-b2)
            value /= (1-b1**(n+1)) # m correction
            # value *= np.sqrt(1-b2**(n+1)) # v correction
    gg = gen()
    gg.send(None)
    return universal(grad, x, gg, b1, b2, eps)

@unflatten_optimizer
def momentum_sgd(grad, x, step_size=0.001, moment=0.9):
    m = np.zeros(len(x))
    for n in itertools.count():
        g = grad(x, n)
        yield x, n, g
        m = m * moment + g * (1-moment) * step_size
        x = x - m

@unflatten_optimizer
def sgd(grad, x, step_size=0.001):
    for n in itertools.count():
        g = grad(x, n)
        yield x, n, g
        x = x - g * step_size

