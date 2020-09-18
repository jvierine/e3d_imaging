#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c

def far_field(theta,d,h=100e3,f=233e6):
    lam = c.c/f
    phi=(2.0*n.pi/lam)*d*n.sin(theta)
    return(phi)

def near_field(theta,d,h=100e3,f=233e6):
    lam = c.c/f
    px = h*n.sin(theta)
    py = h*n.cos(theta)
    phi=(2.0*n.pi/lam)*(n.sqrt( (px+d/2.0)**2.0 + py**2.0)-n.sqrt( (px-d/2.0)**2.0 + py**2.0))
    return(phi)

# image angles
theta = n.pi*n.linspace(-2,2,num=1000)/180.0
h=100e3
d=2e3

phi_ff=far_field(theta,d=d,h=h)
phi_nf=near_field(theta,d=d,h=h)

plt.subplot(121)
plt.plot(theta,phi_ff,label="far field $\phi_{f}(\\theta,d,h)$")
plt.plot(theta,phi_nf,label="near field $\phi_{n}(\\theta,d,h)$")
plt.xlabel("Image pixel angle $\\theta$ (deg)")
plt.ylabel("Phase difference between antennas (rad)")
plt.title("h=%1.2f (km) d=%1.2f (km)"%(h/1e3,d/1e3))
plt.legend()
plt.subplot(122)
plt.plot(theta,phi_ff-phi_nf,label="$\phi_f - \phi_n$")
plt.xlabel("Image pixel angle $\\theta$ (deg)")
plt.ylabel("Phase error $\phi_f - \phi_n$ (rad)")
plt.title("h=%1.2f (km) d=%1.2f (km)"%(h/1e3,d/1e3))
plt.legend()
plt.show()
