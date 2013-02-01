from __future__ import division

from sam.sam import *
from sam.music import *

from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *

import argparse
from scipy.io import wavfile
from glob import *
import re


def audio_wav(file, truncate=None):
    #  audio :: |samples| by |channels|
    sample_rate, audio = wavfile.read(file)

    #  keep first channel
    #  keep first 2^31 samples
    if len(audio.shape)==1:
        nSamples, = audio.shape
        nChannels = 1
        audio = audio[:int32(nSamples)]
    else:
        nSamples, nChannels = audio.shape
        audio = audio[:int32(nSamples), 0]

    # consistent times => consistent frequencies
    if truncate:
        print 'truncating %s' % file
        audio = audio[:truncate]

    # it sounds the same, if you now do
    # wavfile.write(filename, sample_rate, audio)

    return audio, sample_rate

def process_wav(file, truncate=None, window_size = 2**12):
    """
    spectrum : |windows| by window_size
    |frequencies| = window_size
    |windows| / 2 * |frequencies| ~ |seconds| * sample_rate
    """ 
    print 'processing %s...' % file
    audio, sample_rate = audio_wav(file, truncate=truncate)

    hanning_window = hanning(window_size)

    nWindows = int32(audio.size/window_size *2) # double <- overlap windows

    spectrum = zeros((nWindows,window_size))

    for i in xrange(0,nWindows-1):
        t = int32(i* window_size/2)
        window = audio[t : t+window_size] * hanning_window # elemwise mult
        spectrum[i,:] = fft(window)
        
    """ not invertible, sounds like noise
    iaudio = zeros((int(nWindows/2)-1, window_size))
    hanning_window[hanning_window==0]=min(hanning_window[hanning_window!=0]/4)
    for i in xrange(0,int(nWindows/2)-1):
        iaudio[i,:] = 1/hanning_window * abs(ifft(spectrum[2*i,:]))
    iaudio.dtype='int16'
    wavfile.write('inverse.wav', sample_rate, iaudio)
    """

    return spectrum, sample_rate


def to_freq(file):
    file = basename(file)

    frequency = re.match('[A-G][#b]?(?P<freq>[0-9]+)', file)
    if frequency:
        frequency = frequency.groupdict()['freq']
        return int(frequency)

    frequency = re.match('[0-9]+', file)
    if frequency:
        frequency = frequency.group()
        return int(frequency)

    note = re.sub("[^ a-zA-Z #b 0-7]", "", file)
    if note:
        return freq(note)

    raise Exception('bad audio filename "%s"' % file)


def train_joint(dataset, window_size = 2**12):
    print
    print 'TRAINING...'
    
    n  = len(flatten(dataset))
    print 'n =', n

    classifier = zeros((n, window_size))
    
    dataset = [sorted(data, key=to_freq) for data in dataset]
#    def instrument(file): 
#    dataset = [instrument(file) for file in flatten(dataset)]
    #  sort filenames by frequency ascending
    
    freqs = flatten(
        [sorted([note_freq(to_freq(file)) for file in data],
                key=snd) 
#                [tuple([ ('(%s) %s' % (instrument(file),k),f) for k,f note_freq(to_freq(file))]) 
         for data in dataset])
    #  keep track of file's pitch
    #  eg 'A440.wav' => 440
    
    truncate = min( show(audio_wav(file)[0].shape[0]) for i,file in enumerate(flatten(dataset)) )
    truncate = 44100 * 10 #TODO fixes white data bug
    print 'truncating at %s samples' % (truncate)

    for i,file in enumerate(flatten(dataset)):
        spec, _ = process_wav(file, truncate=truncate, window_size=window_size)
        
        # normalize to unit vector
        classifier[i,:] = sum(spec,0) / sum(spec)
        #note  sum by default reduces along all dims to scalar
        
#    print
#    print 'freqs...'
#    for freq in freqs: print freq
#    print
    
    print 'DONE TRAINING.'
    print
    return classifier, freqs

