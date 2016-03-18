#%%
import matplotlib.pyplot as plt
import numpy
import os
import math
import scipy.integrate
import skimage.io, skimage.util, skimage.filters

def rescale(signal, minimum, maximum):
    mins = signal.min()
    maxs = signal.max()

    output = (maximum - minimum) * (signal - mins) / (maxs - mins) + minimum

    return output

print "Setting up constants"

signal = numpy.zeros((256, 256)) # This is the size of your system

dx = 1

alpha = 10.0
y = 5.0

Wx = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[1], 1.0)
Wy = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[0], 1.0)
wx, wy = numpy.meshgrid(Wx, Wy)
wx2, wy2 = numpy.meshgrid(Wx * Wx, Wy * Wy)

wx = wx
wy = wy
wx2 = wx2
wy2 = wy2

c1 = 0.125
c2 = 0.383

c11 = 2.31
c12 = 1.09
c44 = 1.57
eps0 = None

xi = (c11 - c12 - 2 * c44) / c44

sigmas = numpy.ndarray(wx.shape)

e1 = numpy.ndarray(wx.shape)
e2 = numpy.ndarray(wx.shape)

ws = numpy.array([wx, wy])

norms = numpy.linalg.norm(ws, 2, axis = 0)

e1 = ws[0] / norms
e2 = ws[1] / norms

def sig(e1, e2):
    sigmas = (1 + \
       2 * xi * (e1 * e1 * e2 * e2)) / \
   ((1 + \
     xi * ((c11 + c12) / c11) * (e1 * e1 * e2 * e2)))

    return sigmas

sigmas = sig(e1, e2)
sigmas[0, 0] = 1

print "Adjusting to V..."

def asig(angles):
    out = numpy.ndarray(len(angles))
    for i, a in enumerate(angles):
        out[i] = sig(math.cos(a), math.sin(a))

    return out

meanSigma = scipy.integrate.quadrature(asig, 0.0, 2 * numpy.pi, tol = 1e-12, rtol = 1e-12, maxiter = 100)

V = sigmas - meanSigma[0] / (2 * numpy.pi)

a = numpy.real(numpy.fft.ifft2(V))
a = numpy.fft.fftshift(a)
plt.imshow(a[128 - 3 : 128 + 4, 128 - 3 : 128 + 4], interpolation = 'NONE');
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches((8, 6))
plt.show()

plt.imshow(a[128 - 7 : 128 + 8, 128 - 7 : 128 + 8], interpolation = 'NONE');
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches((8, 6))
plt.show()
#%%