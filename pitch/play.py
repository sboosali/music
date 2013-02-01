#!/usr/bin/python
from __future__ import division

from music.sam.sam import *
from music.sam.music import *

from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *

import argparse
from scipy.io import wavfile
from glob import *
import re
import time

# # # # # # # # # # # # # # # # # # # # # # # # 
# log cauchy
# # # # # # # # # # # # # # # # # # # # # # # # 

def logcauchy(x, scale=1, sigma=0.5, mu=5):
    return 1 / (x * pi * sigma * (1 + ((log(x) - mu) / sigma)**2) )

scale = 1
mu = 7
sigma = 1/8
#? stickiness = .04

x = arange(1,1800) / scale
y = 1 / (x * pi * sigma * (1 + ((log(x) - mu) / sigma)**2) )

ioff()
plot(y)
show()

"""
ioff()
mus = [-1,0,1,2,3,4,5,6,7]
for i in xrange(len(mus)):
    mu = mus[i]
    x = arange(1,1800) / scale
    y = logcauchy(x, scale=scale, sigma=sigma, mu=mu)

    axes = gca()
    axes.set_title('mu = %s' % mu)
    plot(y)
    show()
"""

# find local maxima (of 1D function)
# local max x  :=  f(x) > f(x+h)  &&  f(x) > f(x-h)
#  unless x is on the boundary of the domain
# one at zero (= silence) and one at positive (= sound)
def local_maxima(f, xs):
    lm = []
    xs = a(xs)
    ys = f(xs)
    for i,y in enumerate(ys[1:-1]):
        xa = xs[i-1]
        x  = xs[i]
        xb = xs[i+1]
        if f(x) >= f(xa) and f(x) >= f(xb):
            lm.append(x)
    return lm

def show(x):
    if type(x)==type([]):
        for elem in x:
            print elem

    return x

scale = 1/2
xs = arange(1,1800) / scale
mus = [5]
for mu in mus:
    print
    print 'mu = %s...' % mu
    lc = lambda x: logcauchy(x, scale=scale, sigma=sigma, mu=mu)
    maxes = local_maxima(lc, x) 
    show(maxes)

# logcauchy(0) = inf    
lc_0 = lc(1e-3)
lc_mid = lc(maxes[-1])
print 'f(0) == %s  >  f(x*) == %s  :=  %s' % (lc_0, lc_mid, lc_0 > lc_mid)

