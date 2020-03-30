#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt

def convolution_matrix(envelope, rmax=300):
    L=len(envelope)
#    N_m=rmax+L
    A = n.zeros([L, rmax], dtype=n.complex64)
    ridx=n.arange(rmax)
    idx=n.arange(L)
    for i in n.arange(L):

        A[i, :] = envelope[ (i-ridx)%L ] 

    return(A)


def prn_code(L):
    """
    pseudorandom code
    """
    code=n.exp(1j*n.random.rand(L)*n.pi*2.0)


def sim_isr(N_pulses=100,N_rep=100,N_tx=2,N_r=200,L=32):
    """ 
    simulate E-region long correlation length ISR echo 
    """
    
    A_re = n.zeros([N_pulses*N_r,N_r*6],dtype=n.float32)
    m_re = n.zeros(N_pulses*N_r,dtype=n.float32)
    A_im = n.zeros([N_pulses*N_r,N_r*6],dtype=n.float32)
    m_im = n.zeros(N_pulses*N_r,dtype=n.float32)

    z00=n.zeros(N_r,dtype=n.complex64)
    z11=n.zeros(N_r,dtype=n.complex64)
    z01=n.zeros(N_r,dtype=n.complex64)        
    
    for i in range(N_pulses):
        
        
        tx0 = n.zeros(N_r,dtype=n.complex64)
        tx1 = n.zeros(N_r,dtype=n.complex64)

        # random phase binary code
        tx0[0:L]=n.exp(1j*n.random.rand(L)*2*n.pi)
        tx1[0:L]=n.exp(1j*n.random.rand(L)*2*n.pi)

        A0=convolution_matrix(tx0,rmax=N_r)
        A1=convolution_matrix(tx1,rmax=N_r)
        
        tx00=n.zeros(N_r,dtype=n.complex64)
        tx11=n.zeros(N_r,dtype=n.complex64)
        tx01=n.zeros(N_r,dtype=n.complex64)
        tx10=n.zeros(N_r,dtype=n.complex64)        
        
        tx00[0:L] = tx0[0:L]*n.conj(tx0[1:(L+1)])
        tx11[0:L] = tx1[0:L]*n.conj(tx1[1:(L+1)])
        
        # cross-correlation
        tx01[0:L] = tx0[0:L]*n.conj(tx1[1:(L+1)]) + tx1[0:L]*n.conj(tx0[1:(L+1)]) # real part of sigma^01
        tx10[0:L] = tx0[0:L]*n.conj(tx1[1:(L+1)]) - tx1[0:L]*n.conj(tx0[1:(L+1)]) # imag part of sigma^01
        
        A00=convolution_matrix(tx00,rmax=N_r)
        A11=convolution_matrix(tx11,rmax=N_r)
        A01=convolution_matrix(tx01,rmax=N_r)
        A10=convolution_matrix(tx10,rmax=N_r)

        Z=n.zeros(A00.shape,dtype=n.float32)
        A_re0=n.hstack((n.real(A00),-n.imag(A00),  n.real(A11),-n.imag(A11),   n.real(A01),-n.imag(A10)))
        A_im0=n.hstack((n.imag(A00), n.real(A00),  n.imag(A11), n.real(A11),   n.imag(A01), n.real(A10)))

        A_re[(N_r*i):(N_r*i+N_r),:]=n.real(A_re0)

        A_im[(N_r*i):(N_r*i+N_r),:]=n.real(A_im0)

        ml=n.zeros(N_r,dtype=n.complex64)
        
        for j in range(N_rep):
            r0=(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0)
            r1=(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0)
            r2=2*(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0)
        
            # make z0 partially correlated with z1, and have a phase shift
            z0=r0+r1
        
            # make z1 partially correlated with z1, and have a phase shift
            z1=r1*n.exp(-1j*n.pi/4.0)+r2

            z0[0:10]=0
            z0[100:140]=0        
            z1[0:50]=0
            z1[140:200]=0

            z00+=z0*n.conj(z0)
            z11+=z1*n.conj(z1)
            z01+=z0*n.conj(z1)                
        
            # simulate measurement
            m0=n.dot(A0,z0)+n.dot(A1,z1) + n.random.randn(N_r)+n.random.randn(N_r)*1j

            ml[0:(len(m0)-1)] += m0[0:(len(m0)-1)]*n.conj(m0[1:len(m0)])/float(N_rep)
        m_re[(N_r*i):(N_r*i+N_r)]=ml.real
        m_im[(N_r*i):(N_r*i+N_r)]=n.imag(ml)
        
    A=n.vstack((A_re,A_im))
    
    m=n.concatenate((m_re,m_im))
    var_re=n.mean(n.abs(m_im)**2.0)
    var_im=n.mean(n.abs(m_im)**2.0)    

    print(A.shape)
    print(m.shape)
   # xhat=n.dot(Sinv,n.dot(n.transpose(A),m.flatten()))
   
    xhat=n.linalg.lstsq(A,m.flatten())[0]
    print(xhat)
    plt.subplot(221)
    plt.plot(xhat[0:200],label="deconv (re)")
    plt.plot(xhat[200:400],label="deconv (im)")
    plt.title("$\sigma_r^{00}$")

    plt.plot(z00.real/(N_rep*N_pulses),label="true (re)")
    plt.plot(z00.imag/(N_rep*N_pulses),label="true (im)")
    plt.legend()
    plt.xlabel("Range (samples)")    
    plt.subplot(223)
    plt.plot(xhat[400:600],label="deconv (re)")
    plt.plot(xhat[600:800],label="deconv (im)")
    plt.title("$\sigma_r^{11}$")    


    plt.plot(z11.real/(N_rep*N_pulses),label="true (re)")
    plt.plot(z11.imag/(N_rep*N_pulses),label="true (im)")
    plt.xlabel("Range (samples)")
    plt.legend()

    plt.subplot(222)
    plt.plot(xhat[800:1000],label="deconv (re)")
    plt.plot(xhat[1000:1200],label="deconv (im)")
    
    plt.plot(z01.real/(N_rep*N_pulses),label="true (re)")
    plt.plot(z01.imag/(N_rep*N_pulses),label="true (im)")
    plt.title("$\sigma_r^{01}$")
    plt.xlabel("Range (samples)")
    plt.legend()
    plt.tight_layout()
    plt.show()
        



sim_isr()
    
