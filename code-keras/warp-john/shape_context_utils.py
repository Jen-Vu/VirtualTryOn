from numpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
from math import atan2
import numpy as np

def dimension_check(arr):
    if arr.ndim == 1:
        arr = arr.reshape((len(arr), 1))
    return arr

def dist2(x,c):
    """
        Euclidian distance matrix
    """
    ## tested, no issue #######

    x = np.asarray(x)
    c = np.asarray(c)
    ncentres = c.shape[0]
    ndata = x.shape[0]
    return np.dot( ones((ncentres, 1)),  ((( power(x,2) ).T) ).sum(axis=0, keepdims=True)).T  + np.dot( ones((ndata, 1)), ((power(c,2)).T).sum(axis=0, keepdims=True)) - np.multiply( 2,(np.dot(x, (c.T))) )


def bookenstain(X,Y,beta):
    """
        Bookstein PAMI89
    
        Article: Principal Warps: Thin-Plate Splines and the Decomposition of Deformations

    """
    X = asmatrix(X)
    Y = asmatrix(Y)

    N = X.shape[0]
    r2 = dist2(X,X)
    r2 = asmatrix(r2)
    K = multiply(r2,log(r2+eye(N,N)))
    P = concatenate((ones((N,1)),X),1)
    L = bmat([[K, P], [P.H, zeros((3,3))]])
    V = concatenate((Y.H,zeros((2,3))),1)

    L[0:N,0:N] = L[0:N,0:N] + beta * eye(N,N)

    invL = linalg.inv(L)

    # L^-1 * v^T = (W | a_1 a_x a_y)^T
    c = invL*(V.H)
    cx = c[:,0]
    cy = c[:,1]
    
    Q = (c[0:N,:].H) * K * c[0:N,:]
    E = mean(diag(Q))
    return cx,cy,E


def hist_cost(BH1, BH2):
    eps = 2.2204e-16

    BH1 = asmatrix(BH1)
    BH2 = asmatrix(BH2) 

    [nsamp1, nbins] = BH1.shape
    [nsamp2, nbins] = BH2.shape

    BH1n=BH1/tile(BH1.sum(axis=1)+eps,(1, nbins))
    BH2n=BH2/tile(BH2.sum(axis=1)+eps,(1, nbins))
    tmp1=tile( transpose( expand_dims(BH1n, axis=2), (0, 2, 1) ),(1, nsamp2, 1) )
    tmp2=tile( transpose( expand_dims(BH2n.H, axis=2), (2, 1, 0) ),(nsamp1, 1, 1) )
    HC=0.5*( ( (tmp1-tmp2)**2 )/(tmp1+tmp2+eps) ).sum(axis=2)

    return HC


def bdry_extract_3(V):
    Vg=V
    H,W,_ = Vg.shape
    #print("H is {}.".format(H))
    #print("W is {}".format(W))
    x_ind = range(W)
    y_ind = range(H)
    m = Vg[:,:,0]
    c = plt.contour(x_ind,y_ind,m,.5)
    coords = c.allsegs[0][0]
    
    
    # always remember to convert to float for gradient calculate to avoid overflowing
    #[G1,G2]=gradient(Vg.squeeze().astype(float));
    G1 = gradient(Vg.squeeze().astype(float), axis =0)
    G2 = gradient(Vg.squeeze().astype(float), axis =1)

    # need to deal with multiple contours (for objects with holes)
    coords = coords.T
    B = coords[:, coords[0]!=0.5]


    npts=B.shape[1]
    t=zeros((npts,1));
    for n in range(npts): 
        x0=int(round(B[0,n]))
        y0=int(round(B[1,n]))

        t[n]=atan2(G2[y0,x0],G1[y0,x0])+pi/2;

    x=B[np.newaxis,0,:].T
    y=B[np.newaxis,1,:].T

    return x,y,t


def get_samples_1(x, y, t, nsamp):
    N = len(x)
    k = 3
    Nstart = np.min([k*nsamp, N])
    ind0 = np.random.randint(Nstart, size = Nstart)
    ind0 = ind0[:Nstart]
    xi = x[ind0]
    yi = y[ind0] 
    ti = t[ind0]

    i_1 = np.concatenate((xi, yi), axis=1)

    d2 = dist2(i_1, i_1)
    d2 = d2 + np.diagflat(np.Inf*np.ones((Nstart, 1)))

    s = True
    while s:

        # find closest pair
        a = np.min(d2,axis=0)
        b = np.argmin(d2, axis=0)
        J = np.argmin(a)

        xi = np.delete(xi, J)
        yi = np.delete(yi, J)
        ti = np.delete(ti, J)

        d2 = np.delete(d2, (J), axis = 0)
        d2 = np.delete(d2, (J), axis = 1)

        if d2.shape[0] == nsamp:
            s = False

        xi = dimension_check(xi)
        yi = dimension_check(yi)
        ti = dimension_check(ti)

    return xi, yi, ti



def sc_compute(Bsamp,Tsamp,mean_dist,nbins_theta,nbins_r,r_inner,r_outer,out_vec):

    Bsamp = np.asmatrix(Bsamp)
    Tsamp = np.asmatrix(Tsamp)

    nsamp=Bsamp.shape[1]
    in_vec=out_vec==0  # print(in_vec) # (100,1)

    # compute r,theta arrays
    r_array=(np.sqrt(dist2(Bsamp.T,Bsamp.T))).real  #print(r_array.shape) # = (100,100)
    #print("r_array is {}".format(r_array) )

    #print("the average of r_array {}".format(np.mean(r_array)))

    part_1 = np.dot( Bsamp[1,:].T, np.ones((1,nsamp)) ) - np.dot( np.ones((nsamp,1)), Bsamp[1,:] )
    part_2 = np.dot( Bsamp[0,:].T, np.ones((1,nsamp)) ) - np.dot( np.ones((nsamp,1)), Bsamp[0,:] )

    #print("part_1 is {}".format(part_1))
    #print("part_2 is {}".format(part_2))
    #print(sum(part_1))
    #print(sum(part_2))

    theta_array_abs = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            theta_array_abs[i,j] = atan2(part_1[i,j], part_2[i,j])

    theta_array_abs=theta_array_abs.T #print(theta_array_abs.shape) # = (100,100)
    #print("theta_array_abs {}".format(theta_array_abs))
    theta_array=theta_array_abs - np.dot( Tsamp.T, np.ones((1,nsamp)) ) # print(theta_array.shape)  # = (100,100)
    #print("theta_array {}".format(theta_array))

    # create joint (r,theta) histogram by binning r_array and
    # theta_array

    # normalize distance by mean, ignoring outliers
    #print(type(r_array))
    #print(type(Bsamp))
    mean_dist = np.nanmean(r_array)
    r_array_n = r_array/mean_dist
    #print("mean_dist {}".format(mean_dist))
    #print("r_array_n {}".format(r_array_n))

    # use a log. scale for binning the distances
    r_bin_edges=np.logspace(np.log10(r_inner),np.log10(r_outer), num=5)
    #print("r_bin_edges {}".format(r_bin_edges))
    r_array_q=np.asmatrix(np.zeros((nsamp,nsamp)))
    for m in range(nbins_r):
        r_array_q=r_array_q+(r_array_n<r_bin_edges[m])

    #print("r_array_q {}".format(r_array_q))
    #print(np.sum(r_array_q))
    fz=np.asarray(r_array_q>0) # flag all points inside outer boundary
    #print("fz {}".format(fz))
    #print(sum(fz))
    # # put all angles in [0,2pi) range
    theta_array_2 = np.remainder(np.remainder(theta_array,2*pi)+2*pi,2*pi)
    #print("theta_array_2 {}".format(theta_array_2))

    # quantize to a fixed set of angles (bin edges lie on 0,(2*pi)/k,...2*pi
    theta_array_q = 1+np.floor(theta_array_2.astype(float)/(np.float(2*pi)/nbins_theta))
    #print("theta_array_q {}".format(theta_array_q))
    # print("shape of theta_array_2 is {}".format(theta_array_2.shape))
    # print("shape of theta_array_q is {}".format(theta_array_q.shape))
    # print("shape of r_array_q {}".format(r_array_q.shape))


    nbins=np.dot(nbins_theta, nbins_r)
    BH=np.zeros((nsamp,nbins))
    for n in range(nsamp):
        #print(np.sum(fz[np.newaxis,n,:]))
        fzn= fz[n,:] > 0 
        #print(fzn.squeeze())
        
        #print(fzn.shape)
        temp_lj = np.zeros((12,5))

        #print("shape of theta_array_q[n,fzn.squeeze()] {}".format(theta_array_q[n,fzn].shape))
        #print("shape of theta_array_q[n,fzn.squeeze()] {}".format(r_array_q[n,fzn].shape))
        theta_r = np.concatenate(( theta_array_q[n,fzn], r_array_q[n,fzn]), axis=0).astype(int)
        for i in range(theta_r.shape[1]):
            temp_lj[theta_r[0,i]-1, theta_r[1,i]-1] +=1
        BH[n,:]=temp_lj.T.reshape(1, 60)

    #print(BH.shape)
    return BH, mean_dist


def hist_cost_2(BH1, BH2):
    nsamp1, nbins = BH1.shape
    nsamp2, nbins = BH2.shape

    BH1n = BH1/np.tile(np.sum(BH1, axis=1, keepdims=True) + np.finfo(float).eps, (1, nbins))
    BH2n = BH2/np.tile(np.sum(BH2, axis=1, keepdims=True) + np.finfo(float).eps, (1, nbins))

    tmp1 = np.tile( np.transpose( np.expand_dims(BH1n, axis=2), (0, 2, 1) ), (1, nsamp2, 1) )
    tmp2 = np.tile( np.transpose( np.expand_dims(BH2n.T, axis=2), (2, 1, 0) ), (nsamp1, 1, 1) )
    HC=0.5*np.sum( ((tmp1-tmp2)**2)/(tmp1+tmp2+np.finfo(float).eps), axis=2)

    return HC



















