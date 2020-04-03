#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import itertools

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


def sim_isr(N_pulses=128, # number of transmit pulses in the group
            N_rep=50,   # how many times to we cycle through the pulse group and avg
            N_tx=4,      # number of transmitters
            N_r=200,     # number of range gates
            L=16,        # number of bauds in each code
            SNR_1=10.0,  # signal to noise ratio for one transmitter (full array)
            plot=False,
            corr=1.0):   # how correlated are transmit-receive paths
    """ 
    Simulate E-region long correlation length ISR echo 
    SNR_1 is SNR with one transmitter
    """

    n_fun=N_tx + N_tx*(N_tx-1)/2
    
    A_re = n.zeros([N_pulses*N_r,N_r*n_fun*2],dtype=n.float32)
    m_re = n.zeros(N_pulses*N_r,dtype=n.float32)
    A_im = n.zeros([N_pulses*N_r,N_r*n_fun*2],dtype=n.float32)
    m_im = n.zeros(N_pulses*N_r,dtype=n.float32)
    
    # cross-correlations
    zcc=n.zeros([n_fun,N_r],dtype=n.complex64)

    # cc indices
    txidx=n.arange(N_tx)
    l=list(itertools.combinations(txidx,2))
    for txi in n.arange(N_tx)[::-1]:
        l=[(txi,txi)]+l
    print(l)
    n_cc=len(l)
    
    for i in range(N_pulses):
        tx=n.zeros([N_tx,N_r],dtype=n.complex64)
        codes=n.exp(1j*n.random.rand(L*N_tx)*2.0*n.pi)
        codes.shape=(N_tx,L)
        tx[:,0:L]=codes
        
        # these are used to simulate the ISR
        FAs=[]
        for txi in range(N_tx):
            FAs.append(convolution_matrix(tx[txi,:],rmax=N_r))

        ltx=n.zeros(N_r,dtype=n.complex64)
        Aii=[]
        Aij=[]
        Aji=[]
        A_re0=[]
        A_im0=[]
        for ci,cc in enumerate(l):
            if cc[0] == cc[1]:
                ltx[0:L]=tx[cc[0],0:L]*n.conj(tx[cc[1],1:(L+1)])
                A_ii=convolution_matrix(ltx,rmax=N_r)
                A_re0.append(n.real(A_ii))
                A_re0.append(-n.imag(A_ii))
                A_im0.append(n.imag(A_ii))
                A_im0.append(n.real(A_ii))
            else:
                ltx[0:L]=tx[cc[0],0:L]*n.conj(tx[cc[1],1:(L+1)])+tx[cc[1],0:L]*n.conj(tx[cc[0],1:(L+1)])
                A_ij=convolution_matrix(ltx,rmax=N_r)
                
                ltx[0:L]=tx[cc[0],0:L]*n.conj(tx[cc[1],1:(L+1)])-tx[cc[1],0:L]*n.conj(tx[cc[0],1:(L+1)])
                A_ji=convolution_matrix(ltx,rmax=N_r)
                
                A_re0.append(n.real(A_ij))
                A_re0.append(-n.imag(A_ji))

                A_im0.append(n.imag(A_ij))
                A_im0.append(n.real(A_ji))
                
        A_re_row = n.hstack(A_re0)
        A_im_row = n.hstack(A_im0)

        A_re[(N_r*i):(N_r*i+N_r),:]=n.real(A_re_row)

        A_im[(N_r*i):(N_r*i+N_r),:]=n.real(A_im_row)

        ml=n.zeros(N_r,dtype=n.complex64)
        
        for j in range(N_rep):
            # partially correlated
            r_common=n.sqrt(corr)*(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0)

            zs=[]
            m=n.zeros(N_r,dtype=n.complex64)
            for txi in range(N_tx):
                #z.append((n.random.randn(N_r)+n.random.randn(N_r)*1j)/2.0 + r_common)
                z=n.sqrt((1.0-corr))*(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0) + r_common
                zs.append(z)
                m+=n.dot(FAs[txi],z)

            # add noise. more transmitters -> more receiver noise
            # signal power = 1
            # noise power = (1/SNR_1)
            # SNR = SNR_1/N_tx
            m+= n.sqrt(N_tx/SNR_1)*(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0)
            
            # direct CC estimate
            for ci,cc in enumerate(l):
                r0=(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0)
                r1=(n.random.randn(N_r)+n.random.randn(N_r)*1j)/n.sqrt(2.0)
                zcc[ci,:]+=(zs[cc[0]]+r0)*n.conj(zs[cc[1]]+r1)
                
            ml[0:(len(m)-1)] += m[0:(len(m)-1)]*n.conj(m[1:len(m)])
            
        ml=ml/float(N_rep)        
        m_re[(N_r*i):(N_r*i+N_r)]=n.real(ml)
        m_im[(N_r*i):(N_r*i+N_r)]=n.imag(ml)
        
    A=n.vstack((A_re,A_im))    
    m=n.concatenate((m_re,m_im))
    
    var_re=n.mean(n.abs(m_im)**2.0)
    var_im=n.mean(n.abs(m_im)**2.0)    

    Sinv=n.linalg.inv(n.dot(n.transpose(A),A))
    xhat=n.linalg.lstsq(A,m.flatten())[0]

    zcc=zcc/float(N_rep*N_pulses)

    cc_est=n.zeros(N_r)
    ws=0.0
    for ci,cc in enumerate(l):
        if cc[0] == cc[1]:
            cc_est+= xhat[(ci*2*N_r):(ci*2*N_r+N_r)]
            ws+=1.0
        else:
            cc_est +=2*xhat[(ci*2*N_r):(ci*2*N_r+N_r)]
            ws+=2.0
        
            
    
    est_var=n.var(cc_est[0:100]/ws)
    
    if plot:
        plt.subplot(221)
        for ci,cc in enumerate(l):
            plt.plot(zcc[ci,:].real,label=ci)
        plt.legend()
        plt.ylim([-1,2])
    
        plt.subplot(222)
        for ci,cc in enumerate(l):
            plt.plot(zcc[ci,:].imag,label=ci)
        plt.legend()
        plt.ylim([-1,2])

        plt.subplot(223)
        plt.plot(cc_est/ws)

        plt.title(n.var(cc_est[0:100]/ws))
        plt.legend()
        plt.ylim([-1,2])
    
        plt.subplot(224)
        for ci,cc in enumerate(l):
            plt.plot(xhat[(ci*2*N_r+N_r):(ci*2*N_r+2*N_r)],label=ci)
        plt.legend()
        plt.ylim([-1,2])
        plt.show()
    return(est_var)

for i in range(1,5):
    ev=sim_isr(N_tx=i,SNR_1=0.1,N_pulses=16*2,L=4*2)
    print("N_tx %d var %1.5f"%(i,ev))
    
