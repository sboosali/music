#!/usr/bin/python

# matrix multiplication
#  numpy 'dot(A,B)'  is  matlab 'A*B'
# elemwise multiplication
#  numpy 'A*B'  is  matlab 'A.*B'

from __future__ import division

from pitch import *

from numpy import *
from numpy.fft import fft
from numpy.linalg import pinv
from matplotlib.pyplot import *
import argparse
from scipy.io import wavfile
from scipy.stats import norm

from sam.sam import *

"""
from compmusic.pitch.detect import *

dataset = [glob('train/piano/*.wav')]
how = 'nmf'
file = 'chord.wav'

#sorted([int(f.split('/')[-1].split('.')[0][1:]) for f in dataset])
A, freqs = train_joint(dataset)
B, sample_rate = process_wav(file)

_A = zeros(A.shape)
for i in xrange(A.shape[0]): 
    in_max = argmax(half(A[i,:]))
    out_max = max(half(A[i,:]))
    _A[i, in_max] = out_max
x,a = pitch_nmf(_A, B)
d2(x, freqs, sample_rate, window_size, title=how)
"""

ROW = 0
COL = 1

OUT_DIR = 'out'
IMAGE_DIR = 'images'
save = False

def stats(A):
    print
    print 'min  = %.3f' % A.min()
    print 'mean = %.3f' % A.mean()
    print 'std  = %.3f' % A.std()
    print 'max  = %.3f' % A.max()

pitch_detection_algorithms = ['nmf','pinv', 'gd'] # enum
def is_pitch_detection_algorithm(s):
    if s in pitch_detection_algorithms: return s
    else: raise ValueError()

dataset_dirs = ['all', 'white', 'black', 'piano','cello', 'live'] # enum
def is_dataset(s):
    if s in dataset_dirs: return s
    else: raise ValueError()    

if __name__=='__main__':
    # cli
    p=argparse.ArgumentParser(description="Pitch Detection")
    p.add_argument('file', type=str)
    p.add_argument('-how', type=is_pitch_detection_algorithm,
                   default='nmf',
                   help='nmf | pinv | gd')
    p.add_argument('-data', type=is_dataset, #nargs='*', #rets list
                   default='white',
                   help='piano | white | black | cello | all')
    p.add_argument('-ioff', action='store_true', dest='ioff', help='ioff = interact with plot (ion = interact with program)')
    
    args = p.parse_args()
    file = args.file
    how  = args.how
    data = args.data
    
    # train classifier
    piano  = [glob('train/piano/*.wav')]
    white  = [glob('train/white/*.wav')]
    black  = [glob('train/black/*.wav')]
    cello  = [glob('train/cello/*.wav')]
    live  = [glob('train/live/*.wav')]
    everything = cello + white
    dataset = {'all' : everything,
               'piano' : piano,
               'white' : white,
               'black' : black,
               'cello' : cello,
               'live' : live,
               }[data]

else:
    #dataset = [glob('train/octave/*.wav')]
    #file = 'chord.wav'
    #how = 'nmf'
    pass

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Pitch Detectors

# pseudoinverse solution
# solve Ax=b for x given A,b
# x : (d,1) = notes
# b : (d',1) = audio (known from input)
# A : (d',d) = instrument/note overtone/undertone profile (known from training)
# d = dimensionality of hidden var
# d' = dimensionality of observed var
# eg d = 8 = |keys in 1 piano 8ve| . d' = 4096 = |linear bins of audible pitches|
def pitch_pinv(classifier, spectrum):

    n, _ = spectrum.shape
    A  = classifier
    Ai = pinv(A)
    B  = spectrum / sum(spectrum)
    X  = zeros((n, classifier.shape[0]))
    
    print
    print 'PINV...'
    print 'A', A.shape
    print 'Ai', Ai.shape
    print 'X', X.shape
    print 'B', B.shape
    X = dot( B, Ai )
    
    params = {}
    iter_diffs = []
    return t(X), Ai, params, iter_diffs


# nmf solution (ie with multiplicative update)
# solve Ax=b for x
#  where x : nonnegative
def pitch_nmf(classifier, spectrum, iters=50):

    n, _ = spectrum.shape
    d, window_size = classifier.shape
    A = t(classifier)
    X = (1/d) * ones((d, n))
    B = t(spectrum / sum(spectrum))

    # ignore last sample
    X = X[:,:-1]
    B = B[:,:-1]
    n = n-1

    print
    print 'NMF...'
    print '|notes|', d           #eg 8
    print 'sam/win', window_size #eg 4096
    print '|windows|', n  #eg 119
    print 'A', A.shape #eg 4096, 8
    print 'X', X.shape #eg 8, 119
    print 'B', B.shape #eg 4096, 119
    
    # jointly solve AX=B forall samples
    # multiplicative update with euclidean distance
    for i in xrange(iters): # until convergence
        print i

        numerX = mul( t(A), B )    #: 8,1024 * 1024,59
        denomX = mul( t(A), A, X ) #: 8,1024 * 1024,8 * 8,59
        X = X * numerX / denomX    #: 8,59

        # independently solve Ax=b forall samples
        #for i in xrange(n):
            #numerX = mul( t(A), B[:,i] )      #: 8,1024 * 1024,1
            #denomX = mul( t(A), A, X[:,i] )   #: 8,1024 * 1024,8 * 8,1
            #X[:,i] = X[:,i] * numerX / denomX #: 8,1
        

        # learn A
        #numerA = mul( B, t(X) )    #: 1024,59 * 59,8
        #denomA = mul( A, X, t(X) ) #: 1024,8 * 8,59 * 59,8
        #A = A * numerA / denomA    #: 1024,8
        
        #X = _X


    params = {'iters':iters,
              }

    iter_diffs = []
    return X, A, params, iter_diffs

# gradient descent solution (ie with additive update)
def pitch_gd(classifier, spectrum, iters=25, stepsize=100, eps=1e-12, delta=1e-6, alpha=1e-2):
    n, _ = spectrum.shape
    d, sr = classifier.shape
    A = t(classifier.copy())
    X = (1/d) * ones((d, n))
    B = t(spectrum.copy() / sum(spectrum))
    
    X, B = X[:,:-1], B[:,:-1]
    prior = zeros(X.shape)
    
    print
    print 'GD...'
    for _ in xrange(iters):
        
        # gaussian distance likelihood
        # P( b | A,x ) = N[Ax,1]( b )
        # d/dx ||Ax - b||^2 + sum log x
        # = 2A^T||Ax - b|| + sum 1/x
        likelihood = mul( 2 * t(A), mul(A,X) - B )

        # markov stickiness prior
        # P( x[t] | x[t-1] ) =
        # scaled translated complemented logistic( ||x[t] - x[t-1]||^2 )
        # wolfram alpha
        # d/dx -log( (2*e^(-(x-y)^2)) / (1+e^(-(x-y)^2)) )
        time_diff = X[:,1:] - X[:,:-1]
        stats(time_diff)
        #diff = zeros((d,n+1))
        for (i,j) in ndindex((d, n-3)):
            s = j+1
            # t-1 in 0..n-3
            # (j in 0..n-3)
            # t in 1..n-2
            # t+1 in 2..n-1
            ab = X[i,s] - X[i,s-1]
            bc = X[i,s+1] - X[i,s]
            numer_ab = 2 * ab * exp(ab**2)
            denom_ab = 1 + exp(ab**2)
            numer_bc = 2 * bc * exp(bc**2)
            denom_bc = 1 + exp(bc**2)
            prior[i,s] = - (numer_ab / denom_ab) - (numer_bc / denom_bc)
            
        # additive update
        update = likelihood + alpha * prior
        _X = X - stepsize*update
        
        # project onto feasible subspace
        # project onto nonnegative subspace
        # X >= 0
        # |nonnegative R^d| / |R^d| = (1/2)^d -> nonnegative space is sparse!
        _X[_X < delta] = delta
        # project onto unit simplex = dirichlet support
        # sum X over notes (not time) == 1
        normalizer = sum(X) # collapse rows i.e. (m,n) => (1,n)
        _X = _X / normalizer
        
        # dynamic stepsize
        stepsize = stepsize * 0.9

        # convergence
        iter_diff = abs(sum(_X - X))
        #if iter_diff < eps: return X,A

        X = _X
        
    print 'diff:',iter_diff
    #print normalizer[:]
    params = {'iters':iters,
              'stepsize':stepsize,
              'alpha':alpha,
              'eps':eps,
              'delta':delta
              }
              
    return X, A, params


# gradient descent solution (ie with additive update)
# alpha =
#  1e-6 for bach fugue, bach c major prelude
#  3e-5 for chord, scriabin
# iters =
#  50 is always enough
def pitch_gd(classifier, spectrum,  iters=50, eps=1e-12, delta=1e-6, alpha=1e-6):
    stepsize = iters * 2
    _stepsize = stepsize
    
    scale = 1
    mu = 5.5
    sigma = 1/2
    stickiness = 0.04
    
    n, _ = spectrum.shape
    d, sr = classifier.shape
    A = transpose(classifier.copy())
    X = (1/d) * ones((d, n))
    B = transpose(spectrum.copy() / sum(spectrum))
    
    X, B = X[:,:-1], B[:,:-1]
    prior = zeros(X.shape)
    iter_diffs = zeros(iters)
    
    print
    print 'GD...'
    for iteration in xrange(iters):

        # # # # # # # # # # # # # # # # # # 
        # gaussian distance likelihood
        # P( b | A,x ) = N[Ax,1]( b )
        # d/dx    ||Ax - b||^2 + sum log x
        #  =  2A^T||Ax - b||   + sum 1/x

        likelihood = mul( 2 * transpose(A), mul(A,X) - B )

        # # # # # # # # # # # # # # # # # # 
        # markov stickiness prior
        # P( x[t] | x[t-1] ) = 
        #  log cauchy

        if alpha > 0:
            time_diff = X[:,1:] - X[:,:-1]
            # stats(time_diff)
            # for (i,t) in ndindex((d, n-3)):
            for t in range(n)[1:-2]: # [0 +1 .. n-1 -1 -1]
                Y = (abs(time_diff[:,t-1]) + stickiness) / scale
                numer_Y = sigma * (mu**2 - 2*mu + sigma**2 - 2*(mu - 1)*log(Y) + (log(Y))**2)
                denom_Y = pi*(Y**2) * (mu**2 + sigma**2 - 2*mu*log(Y) + (log(Y))**2)**2
                
                Z = (abs(time_diff[:,t]) + stickiness) / scale
                numer_Z = sigma * (mu**2 - 2*mu + sigma**2 - 2*(mu - 1)*log(Z) + (log(Z))**2)
                denom_Z = pi*(Z**2) * (mu**2 + sigma**2 - 2*mu*log(Z) + (log(Z))**2)**2
                
                # prior[i,t] = P( note i of sample t | note i of sample t-1 )
                prior[:,t] = (numer_Y / denom_Y) + (numer_Z / denom_Z)

            Z = (abs(time_diff[:,0]) + stickiness) / scale
            numer_Z = sigma * (mu**2 - 2*mu + sigma**2 - 2*(mu - 1)*log(Z) + (log(Z))**2)
            denom_Z = pi*(Z**2) * (mu**2 + sigma**2 - 2*mu*log(Z) + (log(Z))**2)**2
            prior[:,0] = (numer_Z / denom_Z)
            
            Y = (abs(time_diff[:,-1]) + stickiness) / scale
            numer_Y = sigma * (mu**2 - 2*mu + sigma**2 - 2*(mu - 1)*log(Y) + (log(Y))**2)
            denom_Y = pi*(Y**2) * (mu**2 + sigma**2 - 2*mu*log(Y) + (log(Y))**2)**2
            prior[:,-1] = (numer_Y / denom_Y)
        
        else:
            prior = 1

        # # # # # # # # # # # # # # # # # # 
        # additive update
        update = likelihood + alpha * prior
        _X = X - stepsize*update

        # # # # # # # # # # # # # # # # # # 
        # project onto feasible subspace
        # project onto nonnegative subspace
        #  X >= 0
        #  |nonnegative R^d| / |R^d| = (1/2)^d  ->  nonnegative space is sparse!
        #
        # project onto unit simplex = dirichlet support
        #  sum X over notes (not time) == 1

        normalize = True
        if normalize:
            _X[_X < delta] = delta
            normalizer = sum(X) # collapse rows i.e. (m,n) => (1,n)
            _X = _X / normalizer
        else:
            _X[_X < 0] = 0

        # # # # # # # # # # # # # # # # # #
        # iteration

        # dynamic stepsize
        stepsize = stepsize * 0.9

        # convergence
        iter_diffs[iteration] = abs(sum(_X - X))
        #if iter_diffs[iteration] < eps: return X,A

        print iteration+1
        print iter_diffs[iteration]

        X = _X
    
    params = {'iters':iters,
              'stepsize':_stepsize,
              'alpha':alpha,
              'eps':eps,
              'delta':delta
              }

    print
    print 'diff =' , iter_diffs[-1]
    for key, val in params.items():
        print '%s = %s' % (key, val)

    return X, A, params, iter_diffs


def viterbi(classifier, spectrum):
    A = transpose(classifier.copy())
    B = transpose(spectrum.copy() / sum(spectrum))
    _, n = B.shape
    _, d = A.shape
    X = (1/d) * ones((d, n))
    
    print 'Viterbi...'
    
    # states
    snd, sil = 0, 1 #'sound', 'silence'
    states = [snd, sil]
    S = len(states)

    # stickiness
    a = 0.99 # = P(silence => silence) # d/(d-1)
    b = 0.90 # = P(sound => sound) # (1/2)**(1/window_rate)
    transition = {snd : {snd : a,
                         sil : 1-a },
                  sil : {snd : 1-b,
                         sil : b }
                  }

    # viterbi
    #  simultaneously solve HMM for each note
    
    V    = zeros((n,S), dtype=float64) # stores probabilities
    path = zeros((n,S), dtype=int32)   # stores indices

    # stickiness transition
    def T(s,r): return prod( [transition[si][ri] for si,ri in zip(s,r)] )
    
    # gaussian emission
    def E(s,b):
        variance = 1
        threshold = variance * 10
        x = { sil : 0 , snd : max(max(b), threshold) }[s] # state => note vector
        return norm.pdf(b, loc=mul(A,x), scale=variance)

    # init viterbi
    s0 = sil
    b0 = B[:,0]
    for i,s in enum(states):
        # s=1 by default. should s=max(b) be the maximum energy?
        #  (at some time b=B[:,t]? over whole signal b=B?) what about silence?
        # s=max(B[t] if max(B[t])>threshold else B)
        #  s=max(B) absolute for low energies. s=max(B[t]) relative for high energies
        V[0, s] = E(b0,s) * T(s0,s)

    # recur viterbi
    for t in xrange(1,n): # for each time in hidden MARKOV MODEL
        x = X[:,t]
        b = B[:,t]
        for i,s in enum(states): # for each state in HIDDEN markov model
            i_argmax = argmax([ V[t-1,r] * T(r,s) for r in states]) # we need {0 1}^d states for d notes x[0]..x[d] per time x

            s_argmax = states[i_argmax] 
            s_max    = V[t-1, s_argmax] * T(s_argmax, s) # most likely previous state given current state

            path[t,i] = i_argmax # save backpointer as viterbi path (with elems, not indices: path[t,s] = r)
            V   [t,i] = s_max * E(s, b) # viterbi path probability * transition * emission

    # walk possible viterbi paths backwards to finds path

    return A,X,{},[]
    

def pitch(classifier, spectrum, how='nmf'):

    try:
        return {'nmf'  : pitch_nmf,
                'pinv' : pitch_pinv,
                'gd'   : pitch_gd,
                }[how](classifier, spectrum)
    except KeyError:
        raise ValueError('\n'.join(["pitch()...", " wants how={'nmf'|'pinv'|'gd'}", " got how=%s" % how]))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Visualization

# Bug#636364: ipython

#Plot
# x-axis = time in seconds
#  window_rate = sample_rate / sample_size * 2
#  j => j / window_rate
# y-axis = pitch as note (frequency in Hz)
#  i => freqs[i]

def d2(x, freqs, sample_rate, window_size, title=how):
    d, n_windows = x.shape

    window_rate = 2 * sample_rate / window_size # windows per second

    axes = gca()
    axes.imshow(x,
                cmap=cm.jet, origin='lower', aspect='auto', interpolation='nearest')
    
    axes.set_title(title)
    axes.get_xaxis().set_major_locator(
        LinearLocator(1 + ceil(n_windows/window_rate)))
    axes.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%ds' % round(x/window_rate)))
    
    axes.get_yaxis().set_major_locator(
                LinearLocator(2*d+1))
    axes.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%s' % (freqs[(y-1)//2][0] if odd(y) else '')))
    
    draw()

 
def d3(Z):
    X = a(r(1,times))
    Y = a(r(1,freqs))
    X, Y = meshgrid(X, Y)
    print X.shape
    print Y.shape
    print Z.shape

    from mpl_toolkits.mplot3d import Axes3D
    fig = figure()
    ax = fig.gca(projection='3d')
    times, freqs = Z.shape
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    draw()


def threshold(x, top=10): # cutoff at brightest 'top' percentage
    # NOTE wtf!?
    #  i ran this code three times within several seconds, it gave 3 diff plots, only the 3rd looked like the run a few minutes ago.
    #  klduge: change file and remove pyc
    x = (x-x.min())/(x.max()-x.min()) # normalize to 0 min
    top_percentile = sorted(flatten(x.tolist()), reverse=True)[int(x.size*top/100)-1] # sort desc
    dullest = x < top_percentile
    x[dullest] = 0
    
    return x

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Main
if __name__=='__main__':

    # # # # # # # # # # # # # # # # 
    # Params
    # 2^12 samples/window / 44100 samples/second = 10 windows/second
    # window size of 2^12 ~ sample of 100ms
    # attack length = ? ms
    window_size = 2**12
    
    # # # # # # # # # # # # # # # # 
    # Pitch Detector

    # classifier : |notes| by window_size
    classifier, freqs = train_joint(dataset, window_size=window_size)
    
    # read file
    spectrum, sample_rate = process_wav(file, window_size=window_size)
    
    x,a,params,diffs = pitch(classifier, spectrum, how=how)
    #x = threshold(x,top=5)
    # how nonnegative is argmax P(x)
    print 'nonnegativity =', 1 - (sum(x>=0)/x.size)
    
    # # # # # # # # # # # # # # # # 
    # Image
    
    if not args.ioff:
        ion()
        import time

    title = 'Transcription . %s . %s' % (how, file)
    d2(x, freqs, sample_rate, window_size, title=title)
    
    if save:
        image = 'joint, gd, chord'
        savefig( '%s/%s.png' % (IMAGE_DIR, image), bbox_inches=0 )

    if args.ioff:
        from matplotlib.pyplot import show
        show()

    if not args.ioff:
        time.sleep(60 * 10)

