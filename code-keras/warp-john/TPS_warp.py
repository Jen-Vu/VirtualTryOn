from numpy import *
import matplotlib.pyplot as plt
from shape_context_utils import *
from sklearn.preprocessing import Binarizer
from math import atan2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes as imfill
from scipy.interpolate import griddata
import scipy.io as sio
import hungarian


def dimension_check(arr):
	if arr.ndim == 1:
		arr = arr.reshape((len(arr), 1))
	return arr

def warping(V1, V2):
	################# parameter setting ###################

	display_flag=True
	affine_start_flag=True
	polarity_flag=True
	nsamp=100
	eps_dum=0.25
	ndum_frac=0.25      
	mean_dist_global=[]
	ori_weight=0.1
	nbins_theta=12
	nbins_r=5
	r_inner=0.125
	r_outer=2 
	tan_eps=1.0
	n_iter=6
	beta_init=1
	r=1
	w=4

	################## image loading #######################

	#V1_orig = plt.imread('/Users/liujin/Desktop/mask_0.jpeg') #print(V1_orig.shape) = (128, 128)
	#V2_orig = plt.imread('/Users/liujin/Desktop/mask_4.jpeg') #print(V1_orig.dtype) = unit8

	V1 = V1.squeeze() #print(V1.shape) = (128, 128)
	V2 = V2.squeeze() #print(v1.dtype) = unit8
	print(V1.shape)

	binarizer1 = Binarizer(threshold=0.5).fit(V1)
	V1 = binarizer1.transform(V1) #print(V1.shape) = (128, 128) #print(v1.dtype) = unit8
	binarizer2 = Binarizer(threshold=0.5).fit(V2)
	V2 = binarizer2.transform(V2)

	V1 = imfill(V1)
	V2 = imfill(V2)

	V1 = expand_dims(asarray(V1), axis=2) #print(V1.shape) = (128, 128, 1) #print(v1.dtype) = unit8
	V2 = expand_dims(asarray(V2), axis=2)

	V1 = V1.astype(float) #print(V1.shape) = (128, 128, 1) #print(v1.dtype) = float64
	V2 = V2.astype(float)

	N1, N2, _ = V1.shape
	print("N1 is {}".format(N1))


	################# edge detection ########################

	x2, y2, t2 = bdry_extract_3(V2)
	nsamp2 = len(x2)
	if nsamp2 >= nsamp:
		x2, y2, t2 = get_samples_1(x2, y2, t2, nsamp)
	else:
		print("error: shape #2 does not have enough samples")
	Y = np.concatenate((x2, y2), axis=1)

	x1, y1, t1 = bdry_extract_3(V1)
	nsamp1 = len(x1)
	if nsamp1 >= nsamp:
		x1, y1, t1 = get_samples_1(x1, y1, t1, nsamp)
	else:
		print("error: shape #1 does not have enough samples")
	X = np.concatenate((x1, y1), axis=1)

	# plt.plot(x2, y2,'r+')
	# axes = plt.gca()
	# axes.set_xlim(0,100)
	# axes.set_ylim(128,0)
	# plt.show()

	# plt.plot(x1, y1,'r+')
	# axes = plt.gca()
	# axes.set_xlim(0,100)
	# axes.set_ylim(128,0)
	# plt.show()

	##################### up to here, x1 is horizontal, y1 is vertical #####################


	################ compute correspondence ##################
	Xk = X
	tk = t1
	k = True
	signal = True
	 
	ndum = np.round(ndum_frac*nsamp).astype(int) #print(ndum) # = 25

	out_vec_1 = np.zeros((1,nsamp))
	out_vec_2 = np.zeros((1,nsamp))

	while signal:

		BH1,mean_dist_1=sc_compute(Xk.T,zeros((1,nsamp)),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_1)
		BH2,mean_dist_2=sc_compute(Y.T,zeros((1,nsamp)),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_2)

		# from_mat=sio.loadmat("/Users/liujin/Desktop/hist_cost.mat")
		# BH1 = from_mat['BH1']
		# BH2 = from_mat['BH2']
		# mean_dist_1 = from_mat['mean_dist_1']
		# mean_dist_2 = from_mat['mean_dist_2']
		# t1 = from_mat["t1"]
		# t2 = from_mat["t2"]
		# tk = from_mat["tk"]



		if affine_start_flag:
			if k == True:
				lambda_o = 1000
			else:
				lambda_o = beta_init*r**(k-2)
		else:
			lambda_o = beta_init*r**(k-1)

		beta_k = (mean_dist_2**2)*lambda_o
		#print("beta_k is {}".format(beta_k))
		costmat_shape = hist_cost_2(BH1, BH2)
		#print("costmat_shape is {}".format(costmat_shape))

		######################################################################
		theta_diff = np.tile(tk, (1, nsamp)) - np.tile(t2.T, (nsamp, 1))
		#print("theta_diff is {}".format(theta_diff))

		if polarity_flag:
			costmat_theta = 0.5 * (1 - np.cos(theta_diff)) 
		else:
			costmat_theta = 0.5 * (1 - np.cos(2*theta_diff))

		costmat=(1-ori_weight)*costmat_shape + ori_weight*costmat_theta

		#print("costmat is {}".format(costmat))

		#######################################################################

		nptsd = nsamp + ndum
		costmat2 = eps_dum*np.ones((nptsd,nptsd))
		costmat2[:nsamp,:nsamp] = costmat
		#print("costmat2 is {}".format(costmat2))


		#######################################################################

		# m = Munkres()
		# cvec=m.compute(costmat2)
		# ## my processing to take out index 
		# cvec = np.asarray(cvec)
		# print("cvec is {}".format(cvec))
		# cvec = cvec[np.newaxis, :, 1]

		#m = munkres.Munkres()
		#indexes = m.compute(costmat2.tolist())

		# from_mat=sio.loadmat("/Users/liujin/Desktop/costmat2.mat")
		# costmat2 = from_mat['costmat2']
		indexes = hungarian.lap(costmat2)
		indexes = np.asarray(indexes)
		#print(indexes.shape)
		cvec = indexes[np.newaxis,1,:]
		#print("cvec is {}".format(cvec))
		#print("cvec shape is {}".format(cvec.shape))


		# from_mat=sio.loadmat("/Users/liujin/Desktop/cvec.mat")
		# cvec = from_mat['cvec'] -1

		# #print("cvec is {}".format(cvec))

		# nptsd = from_mat["nptsd"]
		# nptsd = int(nptsd)
		# #print("nptsd is {}".format(nptsd))

		# Xk = from_mat["Xk"]
		# #print("Xk is {}".format(Xk))
		# X = from_mat["X"]
		# #print("X is {}".format(X))
		# Y = from_mat["Y"]

		a= np.sort(cvec)
		cvec2 = np.argsort(cvec)
		#print("cvec2 is {}".format(cvec2))

		out_vec_1=cvec2[0, :nsamp]>nsamp
		#print("out_cvec_1 is {}".format(out_vec_1))
		out_vec_2=cvec[0,:nsamp]>nsamp
		#print("out_cvec_2 is {}".format(out_vec_2))


		X2=np.nan*np.ones((nptsd,2))
		X2[:nsamp,:]=Xk
		X2 = X2[cvec[:].squeeze(), :]
		#print("X2 is {}".format(X2))
		X2b=np.nan*np.ones((nptsd,2))
		X2b[:nsamp,:] = X
		X2b = X2b[cvec[:].squeeze(),:]
		#print("X2b is {}".format(X2b))  ## attention
		Y2 = np.nan*np.ones((nptsd,2))
		Y2[:nsamp,:] = Y

		#print("Y2 is {}".format(Y2))
		#print("X2b is {}".format(X2b))
		#print("Y is {}".format(Y))

		ind_good = np.nonzero(np.logical_not(np.isnan(X2b[:nsamp, 1])))
		n_good = size(np.asarray(ind_good))
		#print("n_good is {}".format(n_good))
		X3b = X2b[ind_good,:].squeeze()
		Y3 = Y2[ind_good,:].squeeze()

		#print("X3b is {}".format(X3b))
		#print("Y3 is {}".format(Y3))

		# ########## ##################################################
		# # plt.plot(X2[:,0], X2[:,1],'r+')
		# # axes = plt.gca()
		# # axes.set_xlim(0,100)
		# # axes.set_ylim(128,0)
		# # plt.show()

		# # plt.plot(Y2[:,0], Y2[:,1],'r+')
		# # axes = plt.gca()
		# # axes.set_xlim(0,100)
		# # axes.set_ylim(128,0)
		# # plt.show()

		# plt.plot(X3b[:,0], X3b[:,1],'r+')
		# axes = plt.gca()
		# axes.set_xlim(0,100)
		# axes.set_ylim(128,0)
		# plt.show()

		# plt.plot(Y3[:,0], Y3[:,1],'r+')
		# axes = plt.gca()
		# axes.set_xlim(0,100)
		# axes.set_ylim(128,0)
		# plt.show()


		# from_mat=sio.loadmat("/Users/liujin/Desktop/book.mat")
		# X3b = from_mat['X3b']
		# Y3 = from_mat['Y3']
		# beta_k = from_mat['beta_k']

		cx,cy,E = bookenstain(X3b,Y3,beta_k)

		#print("cx is {}".format(cx))
		#print("cy is {}".format(cy))
		#print("E is {}".format(E))

		########################### bookenstain is the same ####################

		# calculate affine cost 

		A = np.concatenate( (cx[n_good+1:n_good+3,:], cy[n_good+1:n_good+3,:]), axis=1) 
		#print("A is {}".format(A))
		_, s, _= np.linalg.svd(A)
		#print("s is {}".format(s))
		aff_cost=log(s[0]/s[1])
		#print(aff_cost)  

		# calculate shape context cost
		a1 = np.min(costmat, axis=0, keepdims=True)
		a2 = np.min(costmat, axis=1, keepdims=True)
		input_lj = np.asarray([np.nanmean(a1), np.nanmean(a2)])
		sc_cost = np.max(input_lj)


		# warp each coordinate
		fx_aff = np.dot( cx[n_good:n_good+3].T, np.concatenate( (np.ones((1, nsamp)), X.T), axis = 0) )
		d2 = dist2(X3b, X)
		d2[d2<=0]=0
		U = np.multiply(d2, np.log( d2 + np.finfo(float).eps))
		fx_wrp = np.dot( cx[:n_good].T, U)
		fx = fx_aff + fx_wrp

		fy_aff = np.dot( cy[n_good:n_good+3].T, np.concatenate( (np.ones((1, nsamp)), X.T), axis = 0) )
		fy_wrp = np.dot( cy[:n_good].T, U)
		fy = fy_aff + fy_wrp

		Z = np.concatenate((fx, fy), axis=0)
		Z = Z.T


		# apply to tangent
		Xtan = X + np.dot(tan_eps, np.concatenate( (np.cos(t1), np.sin(t1)), axis = 1) )
		fx_aff = np.dot( cx[n_good:n_good+3].T, np.concatenate( (np.ones((1, nsamp)), Xtan.T), axis = 0) )
		d2 = dist2(X3b, Xtan)
		d2[d2<=0]=0

		U = np.multiply(d2, np.log( d2 + np.finfo(float).eps))
		fx_wrp = np.dot( cx[:n_good].T, U)
		fx = fx_aff + fx_wrp

		fy_aff = np.dot( cx[n_good:n_good+3].T, np.concatenate( (np.ones((1, nsamp)), Xtan.T), axis = 0) )
		fy_wrp = np.dot( cy[:n_good].T, U)

		Ztan = np.concatenate((fx,fy), axis=0)
		Ztan = Ztan.T


		len_lj = Ztan.shape[0]
		tk = np.zeros((len_lj, 1))
		for i in range(len_lj):
			tk[i] = atan2(Ztan[i,1]-Z[i,1], Ztan[i,0]-Z[i,0])


		Xk = Z

		if k == n_iter:
			signal = False
		else:
			k = k+1


	# ########################   image warp    ######################################

	x,y = np.mgrid[0:N2,0:N1]

	x = x.reshape(-1, 1)
	#print("x is {}".format(x))
	y = y.reshape(-1, 1)
	#print("y is {}".format(y))
	M = np.size(x)
	fx_aff = np.dot( cx[n_good:n_good+3].T, np.concatenate( (np.ones((1,M)), x.T, y.T), axis = 0) )
	d2 = dist2(X3b, np.concatenate( (x,y), axis = 1))
	fx_wrp = np.dot( cx[:n_good].T,  np.multiply(d2, np.log( d2 + np.finfo(float).eps)) )
	fx = fx_aff + fx_wrp

	#print("fx is {}".format(fx))

	fy_aff = np.dot( cy[n_good:n_good+3].T, np.concatenate( (np.ones((1,M)), x.T, y.T), axis = 0) )
	fy_wrp = np.dot( cy[:n_good].T,  np.multiply(d2, np.log( d2 + np.finfo(float).eps)) )
	fy = fy_aff + fy_wrp

	grid_x, grid_y = np.meshgrid(np.arange(0, N2, 1), np.arange(0, N1, 1))

	fx = np.asarray(fx)
	fy = np.asarray(fy)

	V1m = griddata((fx.T.squeeze(), fy.T.squeeze()), V1[y, x], (grid_x, grid_y), method='nearest')
	V1m = V1m.squeeze()
  	V1m[np.isnan(V1m)] = 0

 	binarizer = Binarizer(threshold=0.5).fit(V1m)
	V1m = binarizer.transform(V1m) 

	plt.imshow(V1m.squeeze())
	plt.show()
	# fz=find(isnan(V1w)); V1w(fz)=0;
	return V1m


