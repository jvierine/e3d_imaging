#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import itertools
import scipy.misc as sm

def read_coords(plot=False):
#    f=file("e3d_core.txt")
    f=file("e3d_all.txt")
#    f=file("e3d_core_outliers.txt")
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

def window_function(x,y,x0,y0,L=100.0,a=1.0,lancsoz=False):
    """
    2d cos^2 window of width L
    """
    dx=n.abs(x-x0)/(L)
    dy=n.abs(y-y0)/(L)
    
    I=n.zeros(x.shape,dtype=n.float32)
    # Lancsoz:
    if lancsoz:
        idx=n.where( (dx<a) & (dy<a) )
        I[idx]=n.sinc(dx[idx])*n.sinc(dx[idx]/a)*n.sinc(dy[idx])*n.sinc(dy[idx]/a)
    else:
        idx=n.where( (dx<1.0) & (dy<1.0) )
        I[idx]=(n.cos(n.pi*dx[idx]/2.0)**2.0)*n.cos(n.pi*dy[idx]/2.0)**2.0
    return(I)

class radar_image:
    def __init__(self,n_x=21,width=1e3,height=100e3,lam=1.3):
        """
        width = width of the image in meters
        n_x = number of pixels
        """
        self.n_x=n_x
        self.n_par=n_x*n_x
        self.width=width
        self.pixels=n.zeros([n_x,n_x],dtype=n.float32)
        self.lam=lam
        
        # pixel width
        self.sd = 1.0*width/n_x

        # setup 
        self.pixels[int(n_x/2),:]=1.0
        self.pixels[:,int(n_x/2)]=1.0        

        # center points
        w=width/n_x
        x=n.arange(n_x)*w + w/2.0 - width/2.0
        y=n.arange(n_x)*w + w/2.0 - width/2.0        

        xx,yy=n.meshgrid(x,y)
        self.c_x=xx
        self.c_y=yy
        self.c_z=n.repeat(height,n_x*n_x)
        self.c_z.shape=(n_x,n_x)

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

        self.pixels=n.exp(-(0.5/(width/4.0)**2.0)*(self.c_x**2.0 + self.c_y**2.0))

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

def invert_basis_fun_image(r0,c,b,A0,IMG,L=110.0,err_std0=0.01,svd=True,svd_reg=1000.0):
    
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.pcolormesh(r0.c_x,r0.c_y,IMG,cmap="gray")
    plt.xlabel("E-region East-West position (m)")
    plt.ylabel("E-region North-South position (m)")    
    plt.title("True")
    plt.colorbar()


    n_w=int(n.round(r0.width/L))
    x=n.arange(n_w)*L + L/2.0  - r0.width/2.0
    y=n.arange(n_w)*L + L/2.0 - r0.width/2.0

    r0.pixels[:,:]=0.0
    A1=n.zeros([A0.shape[0],n_w*n_w],dtype=n.complex64)

    # calculate the theory matrix
    # We use Lancsoz basis functions to allow forming the image
    # Each parameter of the model is the multiplier for the basis function.
    par_idx=0
    basis_funs=n.zeros([n_w*n_w,r0.n_x,r0.n_x],dtype=n.float32)
    for xi in x:
        for yi in y:
            I=window_function(r0.c_x,r0.c_y,x0=xi,y0=yi,L=L,a=2.0)
#            plt.pcolormesh(I)
 #           plt.colorbar()
  #          plt.show()
            r0.pixels=I
            basis_funs[par_idx,:,:]=I
            m_sim=n.dot(A0,r0.pixels.flatten())
            print("%d"%(par_idx))
            A1[:,par_idx]=m_sim
            par_idx+=1
            
    
    r0.pixels=n.copy(IMG)
    m_sim=n.dot(A0,r0.pixels.flatten())
    # simulate errors
    mean_pwr=n.mean(n.abs(m_sim)**2.0)
    err_std=n.sqrt(mean_pwr)*err_std0
    err=(n.random.randn(len(m_sim))/n.sqrt(2.0) + 1j*n.random.randn(len(m_sim))/n.sqrt(2.0))*err_std
    m_sim=m_sim+err
    
    S=(err_std**2.0)*n.linalg.inv(n.dot(n.conj(n.transpose(A1)),A1))
    post_err_std=n.sqrt(n.diag(n.real(S)))
    post_err_std.shape=(n_w,n_w)
    plt.subplot(222)
    plt.pcolormesh(x,y,post_err_std,cmap="gray")
    plt.xlabel("E-region East-West position (m)")
    plt.ylabel("E-region North-South position (m)")
    plt.title("Error std")
    plt.colorbar()

    if svd:
        u,s,vh=n.linalg.svd(A1)
        sinv=n.diag(s/(s**2.0+svd_reg))

        #    
        print(u.shape)
        print(len(sinv))
        print(vh.shape)
        print(len(m_sim))
        sp=n.zeros([A1.shape[1],A1.shape[0]],dtype=n.complex64)
        sp[:sinv.shape[0],:sinv.shape[0]]=sinv
        print(sp.shape)
        VS=n.dot(n.conj(n.transpose(vh)),sp)
        xhat=n.dot(n.dot(VS,n.conj(n.transpose(u))),m_sim)
    else:
        xhat=n.linalg.lstsq(A1,m_sim)[0]
#    plt.plot(s)
 #   plt.show()
    

    I0=n.zeros([r0.n_x,r0.n_x],dtype=n.float32)
    for i in range(len(xhat)):
        I0+=xhat[i].real*basis_funs[i,:,:]

    plt.subplot(223)
    plt.pcolormesh(r0.c_x,r0.c_y,I0,cmap="gray")
 #   xx,yy=n.meshgrid(x,y)
#    plt.plot(xx.flatten(),yy.flatten(),".",color="green")
    plt.xlabel("E-region East-West position (m)")
    plt.ylabel("E-region North-South position (m)")
    plt.title("Estimate $\Delta h=%1.0f$ (m)"%(L))
    plt.colorbar()

    plt.subplot(224)
    plt.pcolormesh(r0.c_x,r0.c_y,I0-IMG,cmap="gray")
    plt.xlabel("E-region East-West position (m)")
    plt.ylabel("E-region North-South position (m)")
    plt.title("Residual")
    plt.colorbar()

    
    plt.tight_layout()
    plt.show()

def plot_psf(r,b,A0):
    ATA=n.dot(n.transpose(n.conj(A0)),A0)
    psf=ATA[int(ATA.shape[0]/2),:]
    psf.shape=(r.n_x,r.n_x)
    plt.pcolormesh(r.c_x,r.c_y,10.0*n.log10(n.abs(psf)))
    plt.xlabel("Distance (m)")
    plt.title("PSF at 100 km height (dB)")
    plt.ylabel("Distance (m)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    A=sm.imread("aurora.png")
    
    r0 = radar_image(n_x=97,width=2e3)

    IMG=A[:,:,0]#*n.exp(-(0.5/(r0.width/5.0)**2.0)*(r0.c_x**2.0 + r0.c_y**2.0))
    IMG=n.array(IMG,dtype=n.float32)
    IMG=IMG/n.max(IMG)
    
    c=read_coords()
    b=baselines(c)
    # create theory matrix for high resolution image
    A0=create_theory_matrix_nearfield(r0,c,b)

#    plot_psf(r0,b,A0)
    
    Ls=n.arange(40.0)*10.0 + 50.0
    for L in Ls:
        invert_basis_fun_image(r0,c,b,A0,IMG,L=L)        



    
    
