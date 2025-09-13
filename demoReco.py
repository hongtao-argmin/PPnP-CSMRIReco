#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reco. demo in pytorch 
Using Normalization-equivanent DRUNET as the denoiser.
@author: hongtao
"""
import torch
import torchkbnufft as tkbn
import numpy as np
import optalg_Tao_Pytorch as opt # package for optimization algorithms
import scipy.io
import matplotlib.pyplot as plt
import os
from models.network_dncnn import *
from models.fdncnn import *
from models.drunet import *
#=========================================================================
# add arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-im_type', dest='im_type', type=str,default = 'Brain')
parser.add_argument('-trj_type', dest='trj_type', type=str,default = 'Spiral')
parser.add_argument('-sigma_noise', dest='sigma_noise', type=float,default = 0.1)
parser.add_argument('-cuda_index', dest='cuda_index', type=str,default = '0')
parser.add_argument('-im_index', dest='im_index', type=str,default = '1')
parser.add_argument('-MaxIter', dest='MaxIter', type=int,default = 200)
parser.add_argument('-noise_level', dest='noise_level', type=float,default = 1e-3)
parser.add_argument('-MaxCG_Iter', dest='MaxCG_Iter', type=int,default = 20)
parser.add_argument('-NetworkType', dest='NetworkType', type=str,default = 'DRUNet')
parser.add_argument('-model_type', dest='model_type', type=str,default = 'norm-equiv')
parser.add_argument('-verbose', dest='verbose', default = True)
parser.add_argument('-isSave', dest='isSave', default = False)
parser.add_argument('-iSmkdir', dest='iSmkdir', default = True)
args = parser.parse_args()
cuda_index = args.cuda_index
trj_type = args.trj_type #'Radial'#'Spiral' #
im_type = args.im_type #'Knee' # 
im_index = args.im_index# choose which test image
sigma_noise = args.sigma_noise # the noise level for trainning the denoiser
MaxIter = args.MaxIter # maximal number of iterations to recover the image
noise_level = args.noise_level
MaxCG_Iter = args.MaxCG_Iter # the number of iteration for CG
NetworkType = args.NetworkType #'DnCNN' #'FDnCNN'
model_type = args.model_type#'scale-equiv' #'ordinary' #
verbose = args.verbose
isSave = args.isSave # whether save the reconstructed images
iSmkdir = args.iSmkdir
#------------------------------------------------------------------
device = torch.device('cuda:'+cuda_index)
# load the denoising network
savedmodelfolder = '/PredPnP/TrainModel/'
if im_type == 'Brain':
    modelName = 'norm-equivMSECh1_20240320_SE_DRUCNNBrainDenoise_sigma{}.pth'.format(sigma_noise)
elif im_type == 'Knee':
    modelName = 'norm-equivMSECh1_20240320_SE_DRUCNNKneeDenoise_sigma{}.pth'.format(sigma_noise)

if NetworkType == 'DnCNN':
    model = DnCNN(in_nc=1, out_nc=1).to(device)
elif NetworkType == 'FDnCNN':
    model = FDnCNN(blind=True,mode = model_type,in_nc=1, out_nc=1).to(device)
elif NetworkType == 'DRUNet':
    model = DRUNet(blind=True,mode = model_type,in_nc=1, out_nc=1).to(device)   

model_path = (savedmodelfolder+modelName)
model.load_state_dict(torch.load(model_path))
model.eval()

# pick the test image
if im_type == 'Brain':
    filename = 'BrainDeepGT_' + im_index + '.mat'
    data = scipy.io.loadmat('/PredPnP/image/' + filename)
elif im_type == 'Knee':
    filename = 'KneeGT_' + im_index + '.mat'
    data = scipy.io.loadmat('/PredPnP/image/' + filename)

# choose image
im_real = data['im_real']
im_imag = data['im_imag']
im = im_real+1j*im_imag

im = im/np.max(np.abs(im))
im_original = im

# save the GT image.
if trj_type == 'Spiral':
    if im_type == 'Brain':
        folderName = '/PPnP/results/Spiral/DeepBrain'+im_index
    elif im_type == 'Knee':
        folderName = '/PPnP/results/Spiral/Knee'+im_index
    if iSmkdir:
        if not os.path.exists(folderName):
            os.mkdir(folderName)
    trj_file = "data/spiral/trj.npy" # trajectory
    mps_file = "data/spiral/mpsSim32.npy" # sensitivity maps
    mps = np.load(mps_file)
    trj = np.load(trj_file)
    trj = trj[0:-1:6,:,:]
elif trj_type == 'Radial':
    if im_type == 'Brain':
        folderName = '/PredPnP/results/Radial/DeepBrain'+im_index
    elif im_type == 'Knee':
        folderName = '/PredPnP/results/Radial/Knee'+im_index
    if iSmkdir:
        if not os.path.exists(folderName):
            os.mkdir(folderName)
    mps_file = "data/radial/mpsSim32.npy" # sensitivity maps
    mps = np.load(mps_file)
    #1 1 2 3 5 8 13 21 34 55 89
    nspokes = 21
    spokelength = 1024
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(nspokes,spokelength,1))
    ky = np.zeros(shape=(nspokes,spokelength,1))
    ky[0,:,0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[i,:,0] = np.cos(ga) * kx[i - 1,:,0] - np.sin(ga) * ky[i - 1,:,0]
        ky[i,:,0] = np.sin(ga) * kx[i - 1,:,0] + np.cos(ga) * ky[i - 1,:,0]
    trj = np.concatenate((kx,ky),axis=2)

np.save(folderName + '/Real.npy', np.real(im_original))
np.save(folderName + '/Imag.npy', np.imag(im_original))
np.save(folderName + '/Trj.npy',trj)

im_size = im.shape
im = torch.tensor(im).unsqueeze(0).unsqueeze(0).to(torch.complex64)
im = im.to(device)

# trj saved in [(batch) dim NumPoints]
trj_reshape = trj.reshape(1,trj.shape[0]*trj.shape[1],trj.shape[2])
ktraj = torch.tensor(trj_reshape, dtype=torch.float).permute(0,2,1)
ktraj = ktraj.to(device)
# define the NuFFT operator
nufft_ob = tkbn.KbNufft(im_size=im_size,device=device)
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size,device=device)
# set the sensitivity mapping
smaps = torch.tensor(mps).unsqueeze(0) 
smaps = smaps.to(device)

# define the forward model
Ax = lambda x: nufft_ob(x, ktraj, smaps=smaps.to(x))
ATx = lambda x: adjnufft_ob(x,ktraj,smaps=smaps.to(x))
# normalize the forward model
L = opt.Power_Iter(Ax,ATx,im_size,tol = 1e-6,device=device)
L_sr = torch.sqrt(L)
Ax = lambda x: nufft_ob(x, ktraj, smaps=smaps.to(x))/L_sr
ATx = lambda x: adjnufft_ob(x,ktraj,smaps=smaps.to(x))/L_sr
# formulate the measurements
b = Ax(im)
b_m,b_n,b_k = b.shape
torch.manual_seed(2)
noise_real = torch.randn(b_m,b_n,b_k).to(device)
torch.manual_seed(5)
noise_imag = torch.randn(b_m,b_n,b_k).to(device)
b_noise  = b+noise_level*(noise_real+1j*noise_imag)
snr = 10*torch.log10(torch.norm(b)/torch.norm(b_noise-b))
print('The measurements SNR is {0}'.format(snr.cpu().numpy()))

algName = '/PnP_ISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ISTA,psnr_set_PnP_ISTA,CPUTime_set_PnP_ISTA,fixed_PnP_ISTA = \
opt.ISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,save=loc,\
             isPred=False,original=im_original,SaveIter=isSave,verbose = verbose,device=device)

w_pred = lambda x: 2*x-ATx(Ax(x))

algName = '/Pre1_PnP_ISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ISTA_Pre1,psnr_set_PnP_ISTA_Pre1,CPUTime_set_PnP_ISTA_Pre1, fixed_PnP_ISTA_Pre1= \
opt.ISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,save=loc,isPred=True,\
w_pred=w_pred,original=im_original,SaveIter=isSave,verbose = verbose,device=device)


w_pred = lambda x: 4*x-ATx(Ax((10/3)*x))
algName = '/Pre2_PnP_ISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ISTA_Pre2,psnr_set_PnP_ISTA_Pre2,\
CPUTime_set_PnP_ISTA_Pre2,fixed_PnP_ISTA_Pre2 = \
opt.ISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,save=loc,isPred=True,\
w_pred=w_pred,original=im_original,SaveIter=isSave,verbose = verbose,device=device)

algName = '/PnP_Dynamic'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_QNP,psnr_set_PnP_QNP,CPUTime_set_PnP_QNP,fixed_PnP_QNP,gradnorm_PnP_QNP = \
opt.DynamicPnP(MaxIter,Ax,ATx,b_noise,denoiser = model,verbose = verbose,\
save=loc,original=im_original,SaveIter=isSave,device=device)

algName = '/ADMM'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ADMM,psnr_set_PnP_ADMM,CPUTime_set_PnP_ADMM = \
opt.ADMM_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,sigma=1,eta=1,MaxCG_Iter=MaxCG_Iter,save=loc,\
original=im_original,SaveIter=isSave,verbose = verbose,device=device)
