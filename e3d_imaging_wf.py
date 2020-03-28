#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import itertools

def read_coords(plot=False):
    f=file("e3d_all.txt")
    c=[]
    for li,l in enumerate(f.readlines()):
        row=l.split("\t")
        if li > 1 and len(row)>2:
            c.append(n.array([float(row[0]),float(row[1]),float(row[2])]))
    c=n.array(c)
    if plot:
        plt.plot(c[:,1],c[:,0],".")
        plt.show()
    
    return(c)

class radar_image:
    def __init__(self,n_x=21,n_r=100,width=1.5e3,height=100e3,lam=1.3):
        """
        width = width of the image in meters
        n_x = number of pixels
        n_r = render resolution
        """
        self.n_x=n_x
        self.n_par=n_x*n_x
        self.width=width
        self.pixels=n.zeros([n_x,n_x],dtype=n.float32)
        self.lam=lam
        
        # pixel width
        self.sd = 1.0*width/n_x

        # setup 


        # center points for pixels
        w=width/n_x
        x=n.arange(n_x)*w + w/2.0 - width/2.0
        y=n.arange(n_x)*w + w/2.0 - width/2.0

        xx,yy=n.meshgrid(x,y)
        self.pc_x=xx
        self.pc_y=yy
        self.pc_z=n.repeat(height,n_x*n_x)
        self.pc_z.shape=(n_x,n_x)

        self.pcc_x=self.pc.x.flatten()
        self.pcc_y=self.pc.y.flatten()
        self.pcc_z=self.pc.z.flatten()
        
        self.pixels=n.exp(-(0.5/(width/4.0)**2.0)*(self.pc_x**2.0 + self.pc_y**2.0))        
        
        # center points for render pixels
        w=width/n_r
        x=n.arange(n_r)*w + w/2.0 - width/2.0
        y=n.arange(n_r)*w + w/2.0 - width/2.0

        xx,yy=n.meshgrid(x,y)

        
        
        self.c_x=xx
        self.c_y=yy
        self.c_z=n.repeat(height,n_r*n_r)
        self.c_z.shape=(n_r,n_r)

        self.cc_x=self.c_x.flatten()
        self.cc_y=self.c_y.flatten()
        self.cc_z=self.c_z.flatten()
        
        self.tx=n.array([0.0, 0.0, 0.0])

        # calculate unit vectors
        self.ux=(self.cc_x-self.tx[0])
        self.uy=(self.cc_y-self.tx[1])
        self.uz=(self.cc_z-self.tx[2])
        L=n.sqrt(self.ux**2.0+self.uy**2.0+self.uz**2.0)
        self.ux=self.ux/L
        self.uy=self.uy/L
        self.uz=self.uz/L



        up=n.array([0,0,1.0])
        self.zenith_ang=n.arccos((self.ux*up[0] + self.uy*up[1] + self.uz*up[2]))
        self.zenith_ang.shape=(self.n_x,self.n_x)
        
        
    def render(self,n_x=200):
        """ 
        Render with arbitrary resolution
        """ 
        x = n.linspace(-1.1*self.width/2.0,1.1*self.width/2.0,num=n_x)
        y = n.linspace(-1.1*self.width/2.0,1.1*self.width/2.0,num=n_x)

        xx,yy=n.meshgrid(x,y)

        im = n.zeros([n_x,n_x],dtype=n.float32)

        for i in range(self.n_x):
            for j in range(self.n_x):
                Lx=n.sqrt((xx-self.c_x[i,j])**2.0)
                Ly=n.sqrt((yy-self.c_y[i,j])**2.0)
                okx=Lx < self.sd
                oky=Ly < self.sd                
                im+=self.pixels[i,j]*okx*oky*(n.cos((n.pi/2.0)*Lx/self.sd)**2.0 + n.cos((n.pi/2.0)*Ly/self.sd)**2.0)/4.0
        
        return(im,xx,yy)

class baselines():
    """
    Calculate baselines, given antenna coordinates
    """
    def __init__(self,c):
        
        self.n_antennas=c.shape[0]
        self.n_baselines=2*self.n_antennas*(self.n_antennas-1)/2 + self.n_antennas
        
        self.ant_idx=n.arange(self.n_antennas,dtype=n.int)
        self.ap=itertools.combinations(self.ant_idx,2)
        
        self.idx=n.zeros([self.n_baselines,2],dtype=n.int)
        self.pos_a=n.zeros([self.n_baselines,3],dtype=n.float64)
        self.pos_b=n.zeros([self.n_baselines,3],dtype=n.float64)
        self.pos_ab=n.zeros([self.n_baselines,3],dtype=n.float64)            
        
        for i,p in enumerate(self.ap):
            self.idx[2*i,0]=p[0]
            self.idx[2*i,1]=p[1]
            self.pos_a[2*i,:]=c[p[0],:]
            self.pos_b[2*i,:]=c[p[1],:]
            self.pos_ab[2*i,:]=self.pos_b[2*i,:]-self.pos_a[2*i,:]
            self.idx[2*i+1,0]=p[1]
            self.idx[2*i+1,1]=p[0]
            self.pos_a[2*i+1,:]=c[p[1],:]
            self.pos_b[2*i+1,:]=c[p[0],:]
            self.pos_ab[2*i+1,:]=self.pos_b[2*i+1,:]-self.pos_a[2*i+1,:]

        # add zero lags
        i0=2*self.n_antennas*(self.n_antennas-1)/2
        for i in range(self.n_antennas):
            self.pos_a[i+i0,:]=c[i,:]
            self.pos_b[i+i0,:]=c[i,:]
            self.pos_ab[i+i0,:]=n.array([0,0,0])
        
            

def create_theory_matrix_farfield(r,c,b):
    m=n.zeros(b.n_baselines,dtype=n.complex64)
    A=n.zeros([b.n_baselines,r.n_par],dtype=n.complex64)
    
    for i in range(b.n_baselines):
        A[i,:]=n.exp(1j*(2.0*n.pi/r.lam)*(r.ux*b.pos_ab[i,0]+r.uy*b.pos_ab[i,1]+r.uz*b.pos_ab[i,2]))
    return(A)

def create_theory_matrix_nearfield(r,c,b):
    m=n.zeros(b.n_baselines,dtype=n.complex64)
    A=n.zeros([b.n_baselines,r.n_par],dtype=n.complex64)
    
    for i in range(b.n_baselines):        
        # target to rx a
        L1=n.sqrt((r.cc_x-b.pos_a[i,0])**2.0 + (r.cc_y-b.pos_a[i,1])**2.0 + (r.cc_z-b.pos_a[i,2])**2.0)
        
        # target to rx b
        L2=n.sqrt((r.cc_x-b.pos_b[i,0])**2.0 + (r.cc_y-b.pos_b[i,1])**2.0 + (r.cc_z-b.pos_b[i,2])**2.0)        

        A[i,:]=n.exp(1j*(2.0*n.pi/r.lam)*(L1-L2))
        
    return(A)

    
def core():
    c_core=read_coords()
    c_core=c_core[0:109,:]
    b_core=baselines(c_core)

    A_core=create_theory_matrix_farfield(r,c_core,b_core)
    A_core_nf=create_theory_matrix_nearfield(r,c_core,b_core)
    psf_ff=ATA_ff[int(ATA_ff.shape[0]/2),:]
    psf_ff.shape=(r.n_x,r.n_x)
    
    ATA_nf=n.dot(n.conj(n.transpose(A_nf)),A_nf)
    psf_nf=ATA_nf[int(ATA_nf.shape[0]/2),:]
    psf_nf.shape=(r.n_x,r.n_x)
    
    plt.pcolormesh(n.real(psf_ff-psf_nf)/n.abs(psf_nf))
    plt.colorbar()
    plt.show()
    
    ATA_core=n.dot(n.conj(n.transpose(A_core)),A_core)
    psf_core=ATA_core[int(ATA_core.shape[0]/2),:]
    psf_core.shape=(r.n_x,r.n_x)
    
    plt.plot(180.0*r.zenith_ang[r.n_x/2,:]/n.pi,10.0*n.log10(psf_ff[int(r.n_x/2),:]),label="All antennas")
    plt.plot(180.0*r.zenith_ang[r.n_x/2,:]/n.pi,10.0*n.log10(psf_core[int(r.n_x/2),:]),label="Core")
    plt.legend()
    plt.xlabel("Zenith angle (deg)")
    plt.ylabel("PSF Magnitude (dB)")
    plt.show()
    
    plt.pcolormesh(10.0*n.log10(n.abs(psf_ff)))
    plt.colorbar()
    plt.show()
    plt.pcolormesh(10.0*n.log10(n.abs(psf_core)))
    plt.colorbar()
    plt.show()
    print(ATA.shape)

if __name__ == "__main__":

    r0 = radar_image(n_x=101)
    plt.pcolormesh(r0.pixels)
    plt.colorbar()
    plt.show()
    c=read_coords()
    b=baselines(c)

    # simulate measurements
    A0=create_theory_matrix_nearfield(r0,c,b)    
    m_sim=n.dot(A0,r0.pixels.flatten())
    a=n.mean(n.abs(m_sim))
    m_sim=m_sim+0.05*(n.random.randn(len(m_sim))/n.sqrt(2.0)+1j*n.random.randn(len(m_sim))/n.sqrt(2.0))


    for n_pix in range(20):
        r = radar_image(n_x=1+n_pix)
        A1=create_theory_matrix_farfield(r,c,b)
        S=(0.05**2.0)*n.linalg.inv(n.dot(n.transpose(n.conj(A1)),A1))
        print("n_pix %d dr %1.2f std_dev %1.3f"%(n_pix+1,r.width/(n_pix+1),n.mean(n.sqrt(n.diag(S.real)))))
        xhat=n.linalg.lstsq(A1,m_sim)[0]
        xhat.shape=(r.n_x,r.n_x)
        plt.pcolormesh(xhat.real)
        plt.title("Xhat %d,%d"%(n_pix+1,n_pix+1))
        plt.colorbar()
        plt.show()
    
    
