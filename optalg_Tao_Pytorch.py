#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 00:04:30 2023
implement the optimization algorithms in pytorch 
for classical model-based reconstruction
@author: hongtao
"""
import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def Power_Iter(A,AT,im_size,tol = 1e-6,device='cpu'):
    ''' 
    Power iteration to estimate the maximal eigenvalue of AHA.
    '''
    b_k = (torch.randn(im_size)).unsqueeze(0).unsqueeze(0).to(device)
    if AT == None:
        Ab_k = A(b_k+1j*b_k)
    else:
        Ab_k = AT(A(b_k+1j*b_k))
    norm_b_k = torch.norm(Ab_k)
    while True:
        b_k = Ab_k/norm_b_k
        if AT==None:
            Ab_k = A(b_k)
        else:
            Ab_k = AT(A(b_k))
        norm_b_k_1 = torch.norm(Ab_k)
        if torch.abs(norm_b_k_1-norm_b_k)<=tol:
            break
        else:
            norm_b_k = norm_b_k_1
    #b = b_k
    L = torch.vdot(b_k.flatten(),Ab_k.flatten()/torch.vdot(b_k.flatten(),b_k.flatten()))
    return torch.real(L)

def CG_Alg_Handle(x_k,RHS,A,MaxCG_Iter,tol=1e-6):
    r_k = RHS - A(x_k)
    p_k = r_k
    for iter in range(MaxCG_Iter):
        Ap_k = A(p_k)
        alpha_k = torch.vdot(r_k.flatten(),r_k.flatten())/torch.vdot(p_k.flatten(),Ap_k.flatten())
        x_k_1 = x_k+alpha_k*p_k
        if iter<MaxCG_Iter:
            r_k_1 = r_k - alpha_k*A(p_k)
            if torch.norm(r_k_1)<tol:
                break
            beta_k = torch.vdot(r_k_1.flatten(),r_k_1.flatten())/torch.vdot(r_k.flatten(),r_k.flatten())
            p_k_1 = r_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            x_k = x_k_1
    return x_k_1

def PSNR(original, compressed):
    mse = np.mean((np.abs(original - compressed)) ** 2)
    if(mse == 0):  
        return 100
    # compute the scale of the image
    if np.max(np.abs(original))<1.01:
        max_pixel = 1
    else:
        max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr

def denoiseCNN(y_noise,scale,denoiser):
    torch_tempR = torch.real(y_noise)
    torch_tempI = torch.imag(y_noise)
    temp = torch.max(torch.max(torch.abs(torch_tempR)),torch.max(torch.abs(torch_tempI)))
    torch_tempR = torch.real(y_noise)/(temp*scale)
    torch_tempI = torch.imag(y_noise)/(temp*scale)
    torch_temp = torch.cat((torch_tempR,torch_tempI),dim=1)
    with torch.no_grad():
        denoise_out = denoiser(torch_temp)
    temp_R = denoise_out[:,0,:,:].unsqueeze(0)*(temp*scale)
    temp_I = denoise_out[:,1,:,:].unsqueeze(0)*(temp*scale)
    x = (temp_R+1j*temp_I)
    return x

def denoiseCNNMag(y_noise,scale,denoiser):
    torch_mag = torch.abs(y_noise)
    torch_phase = torch.angle(y_noise)
    torch_mag = torch_mag/scale
    with torch.no_grad():
        denoise_out = denoiser(torch_mag)
    temp_mag = denoise_out*scale
    x = temp_mag*torch.exp(1j*torch_phase)
    return x

def ISTA_PnP(num_iters,Ax,ATx,b,Ch = 1, denoiser=None,L=1,isPred = False,\
             w_pred=lambda x: x,save=None,original=None,SaveIter=False,\
verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ISTA PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x)
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
    Ch: 1(default)-one challel denoising only mag; 2(or others)-real and imaginary parts denoising
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_time,lst_fixed
    """

    AHb = ATx(b)
    x = AHb.to(device)
    lst_time  = []
    lst_psnr = []
    lst_fixed = []
    if verbose:
        if isPred:
            pbar = tqdm(total=num_iters, desc="ISTA Pre - PnP",\
            leave=True)
        else:
            pbar = tqdm(total=num_iters, desc="ISTA PnP",\
            leave=True)
    
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    for k in range(num_iters):
        start_time = time.perf_counter()
        gr = ATx(Ax(x))-AHb
        temp = x-w_pred(gr)/L
        if Ch==1:
            x = denoiseCNNMag(temp,1,denoiser)
        elif Ch==2:
            x = denoiseCNN(temp,1,denoiser)
        
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-w_pred(ATx(Ax(x))-AHb)/L,1,denoiser)
        else:
            fixed_temp = x - denoiseCNN(x-w_pred(ATx(Ax(x))-AHb)/L,1,denoiser)
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy())
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())                
            np.save("%s/fixed.npy" % save, lst_fixed)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,np.cumsum(lst_time),lst_fixed

def DynamicPnP(num_iters,Ax,ATx,b,Ch = 1,denoiser = None,L=1,a_k=1,save=None,original=None,SaveIter=False,\
verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with Dynamic PnP:
  ... math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_time
    """
    
    AHb = ATx(b)
    x = AHb.to(device)#torch.zeros_like(AHb)#
    lst_time  = []
    lst_psnr = []
    lst_fixed = []
    lst_gradnorm = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="Dynamic PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    x_old = x
    epsi = 1e-8
    eta = 1
    theta_1 = 2e-6
    theta_2 = 2e2
    beta_stepsize = 0.01
    MaxIterBeta = 100
    for k in range(num_iters):
        start_time = time.perf_counter()
        gr = ATx(Ax(x))-AHb
        if k==0:
            temp_grad = x-gr/L
            gr_old = gr
        else:
            y_k = gr-gr_old/L
            s_k = x-x_old
            x_old = x
            gr_old = gr
            beta = 0
            for iter_beta in range(MaxIterBeta):
                v_k = beta*s_k+((1-beta)*eta)*y_k
                v_k_s_k = torch.real(torch.vdot(v_k.flatten(),s_k.flatten()))
                s_k_s_k = torch.real(torch.vdot(s_k.flatten(),s_k.flatten()))
                v_k_v_k = torch.real(torch.vdot(v_k.flatten(),v_k.flatten()))
                if v_k_s_k/s_k_s_k>=theta_1 and v_k_v_k/v_k_s_k<=theta_2:
                    break
                else:
                    beta = beta+beta_stepsize
            tau_temp = (s_k_s_k/v_k_s_k)
            tau = tau_temp-torch.sqrt(tau_temp**2-s_k_s_k/v_k_v_k)
            if tau<0:
                temp_grad = x_old-gr/L
                D = 1
                Bx_inv = lambda xx: xx
            else:
                H_0 = tau
                D_inv = H_0
                rho = v_k_s_k-H_0*v_k_v_k
                if torch.abs(rho)<=epsi*(s_k_s_k-2*H_0*v_k_s_k+H_0**2*v_k_v_k)*torch.norm(v_k):
                    u = 0
                    u_sign = 0
                    Bx_inv = lambda xx: (a_k*H_0)*xx
                else:
                    temp_1 = s_k-H_0*v_k
                    u_sign = torch.sign(rho)
                    u_inv =  temp_1/torch.sqrt(torch.abs(rho))
                    Bx_inv = lambda xx: ((a_k*H_0)*xx+(u_sign*torch.vdot(u_inv.flatten(),xx.flatten()))*u_inv)
            temp_grad = x_old-Bx_inv(gr_old)
        if Ch == 1:
            x = denoiseCNNMag(temp_grad,1,denoiser)
        else:
            x = denoiseCNN(temp_grad,1,denoiser)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-(ATx(Ax(x))-AHb)/L,1,denoiser)#+mu_corr*x
        else:
            fixed_temp = x - denoiseCNN(x-(ATx(Ax(x))-AHb)/L,1,denoiser)
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy()) 
        lst_gradnorm.append(torch.norm(gr).cpu().numpy())
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
            np.save("%s/fixed.npy" % save, lst_fixed)
            np.save("%s/gradnorm.npy" % save, lst_gradnorm)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,np.cumsum(lst_time),lst_fixed,lst_gradnorm

def ADMM_PnP(num_iters,Ax,ATx,b,Ch = 1,denoiser = None,scale=None,sigma=1,eta=1,MaxCG_Iter = 4,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ADMM PnP:
  .. math:
    \min_x \frac{1}{2sigma^2} \| A x - b \|_2^2 +\phi(x)
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
    sigma: represents the noise level, but one can set it to be one and apply it in the denoiser itself.
    eta: the parameters inside ADMM, has many different ways to choose it but we set it to be 1 following
    Rizwan Ahmad et al. Plug-and-Play Methods for Magnetic Resonance Imaging Using denoisers for image recovery. IEEE SPM.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_time
    """

    AHb = ATx(b)
    lst_time  = []
    lst_psnr = []
    x = AHb
    if verbose:
        pbar = tqdm(total=num_iters, desc="ADMM PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    v_k = AHb
    u_k = torch.zeros_like(v_k)
    Ax_red = lambda xx: ATx(Ax(xx))+(sigma/eta)*xx
    for k in range(num_iters):
        start_time = time.perf_counter()
        if k==0:
            RHS = AHb
        else:
            RHS = AHb+(sigma/eta)*(v_k-u_k)
        x = CG_Alg_Handle(x,RHS,Ax_red,MaxCG_Iter,tol=1e-6)
        z = x+u_k
        # call denoiser
        if Ch == 1:
            v_k = denoiseCNNMag(z,scale,denoiser)
        else:
            v_k = denoiseCNN(z,scale,denoiser)
        u_k = u_k+(x-v_k)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,np.cumsum(lst_time)
