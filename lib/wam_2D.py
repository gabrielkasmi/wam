# -*- coding: utf-8 -*-

# libraries
import torch
import numpy as np
import ptwt
import cv2


def to_numpy(coeffs, dimension, grads=True):
    """
    helper that converts the coefficients 
    to a numpy array
    """

    if dimension==1: # case of sounds
        if grads:
            numpy_coeffs=[c.grad.detach().cpu().numpy() for c in coeffs]
        else:
            numpy_coeffs=[c.detach().cpu().numpy() for c in coeffs]

    if dimension==2: # case of images

        if grads:

            numpy_coeffs=[coeffs[0].grad.detach().cpu().numpy()]
            for coeff in coeffs[1:]:
                numpy_coeffs.append(
                    ptwt.constants.WaveletDetailTuple2d(
                        coeff.horizontal.grad.detach().cpu().numpy(),
                        coeff.vertical.grad.detach().cpu().numpy(),
                        coeff.diagonal.grad.detach().cpu().numpy()
                    )
                )

        else:
            
            numpy_coeffs=[coeffs[0].detach().cpu().numpy()]
            for coeff in coeffs[1:]:
                numpy_coeffs.append(
                    ptwt.constants.WaveletDetailTuple2d(
                        coeff.horizontal.detach().cpu().numpy(),
                        coeff.vertical.detach().cpu().numpy(),
                        coeff.diagonal.detach().cpu().numpy()
                    )
                )

    return numpy_coeffs

class BaseWAM2D():
    def __init__(self,
                 model,
                 wavelet="haar",
                 J=3,
                 device=None,
                 mode="reflect",
                 approx_coeffs=False,
                 normalize_coeffs=True
                 ):
        """
        """
        self.wavelet=wavelet
        self.J=J
        self.mode=mode
        self.approx_coeffs=approx_coeffs
        self.normalize_coeffs=normalize_coeffs

        if device is not None:
            model=model.to(device)
            self.model=model
            self.device=device
        
        else:
            device=next(model.parameters()).device
            self.model=model
            self.device=device


    def __call__(self,x,y,image=True):
        """
        x: should be a tensor of shape [N,C,W,H]
        y: should be a list of shape [N] indicating the 
           **class index** of the images

        approx_coeffs (bool, opt): whether the approximation coefficients 
                                   are included for the visualization
                                   of the reprojection in the pixel domain

        returns a projection on the dyadic wavelet 
        transform of the image, edits an attribution 
        disentangled_scales returning the projection in the pixel domain
        """

        # compute the wavelet transform of the image
        if image:
            coeffs=ptwt.wavedec2(x,self.wavelet,level=self.J,mode=self.mode)

        else:
            coeffs=x

        # require the gradients on the wavelet coefficients
        grad_coeffs=[coeffs[0].requires_grad_()]
        for coeff in coeffs[1:]:
            grad_coeffs.append(
                ptwt.constants.WaveletDetailTuple2d(coeff.horizontal.requires_grad_(),
                coeff.vertical.requires_grad_(),
                coeff.diagonal.requires_grad_())
            )

        # reconstruct the batch from the coefficients with gradients required

        
        img=ptwt.waverec2(grad_coeffs, self.wavelet)
        output = self.model(img.to(self.device)) 
        loss = torch.diag(output[:, y]).mean()
        loss.backward()

        # store the value of the wavelet coefficients
        # the coefficients are not averaged across channels
        self.wavelet_coeffs=to_numpy(grad_coeffs, dimension=2, grads=False)

        # store the value of the gradients of the 
        # wavelet coefficients. The coefficients are not averaged across
        # channels
        self.gradient_coeffs=to_numpy(grad_coeffs, dimension=2, grads=True)

        # reproject the importance in each level for all images in the batch
        self.scales=self.disentangle_scales(self.gradient_coeffs,approx_coeffs=self.approx_coeffs)

        # returns the importance 
        return self.visualize_grad_wam(self.gradient_coeffs)

    def disentangle_scales(self,coeffs, approx_coeffs=False):
        """
        reprojects the important coefficients
        in the pixel domain

        returns an array of size (batch_size, num_levels+1, img_size, img_size)

        coefficients are ordered from the finest to the coarsest scales

        """
        # retrieve the dimensions of the image and 
        # the batch size
        batch_size=coeffs[0].shape[0]
        img_size= int(2 * coeffs[-1].horizontal.shape[-1]) # the image size
                                                           # is twice the size of the 
                                                           # last coefficients

        num_levels=self.J

        if approx_coeffs:
            visualization=np.zeros((batch_size,num_levels+1,img_size,img_size))

        else:
            visualization=np.zeros((batch_size,num_levels,img_size,img_size))
        
        # start by the outermost level
        for i,coeff in enumerate(coeffs[1:][::-1]):

            # retrieve and average the horizontal, diagonal and vertical 
            # components
            horizontal=coeff.horizontal.mean(axis=1)
            diagonal=coeff.diagonal.mean(axis=1)
            vertical=coeff.vertical.mean(axis=1)

            # compute the absolute value of the gradients and normalize
            horizontal=np.abs(horizontal)
            horizontal/=horizontal.max()

            diagonal=np.abs(diagonal)
            diagonal/=diagonal.max()

            vertical=np.abs(vertical)
            vertical/=vertical.max()

            # upsample the coefficients ant plot them
            for img_batch in range(batch_size):

                tmp_horizontal=cv2.resize(horizontal[img_batch], (img_size,img_size), interpolation=cv2.INTER_LINEAR)
                tmp_vertical=cv2.resize(vertical[img_batch], (img_size,img_size), interpolation=cv2.INTER_LINEAR)
                tmp_diagonal=cv2.resize(diagonal[img_batch], (img_size,img_size), interpolation=cv2.INTER_LINEAR)

                visualization[img_batch,i,:,:]=tmp_vertical+tmp_diagonal+tmp_horizontal

        # add the approximation coefficients 
        if approx_coeffs:

            approx=coeffs[0]
            approx=approx.mean(axis=1)
            approx=np.abs(approx)
            approx/=approx.max()

            visualization[img_batch,num_levels,:,:]=cv2.resize(
                approx[img_batch], (img_size,img_size), interpolation=cv2.INTER_LINEAR
            )
                
        return visualization

    def visualize_grad_wam(self,coeffs):
        """
        returns the importance of the wavelet
        coefficients on the wavelet transform
        of the image

        coeffs is a list that contains the 
        wavelet coefficients of the image

        the gradients are averaged across the channels

        returns an array (batch_size,img_size,img_size)
        """

        # retrieve the dimensions of the image and 
        # the batch size
        batch_size=coeffs[0].shape[0]
        img_size= int(2 * coeffs[-1].horizontal.shape[-1]) # the image size
                                                           # is twice the size of the 
                                                           # last coefficients

        visualization=np.zeros((
            batch_size, img_size, img_size
        )) # we average the gradient values across channels

        # add the approximation coefficients on the 
        # upper left corner
        approx=coeffs[0].mean(axis=1)
        approx=np.abs(approx)

        if self.normalize_coeffs:
            approx/=approx.max()

        visualization[:,:approx.shape[1], :approx.shape[2]]=approx #/approx.max()

        # add the approximation coefficients
        for i, coeff in enumerate(coeffs[1:][::-1]):
            
            end_index=int(img_size/2**i)
            start_index=int(img_size/2**(i+1))

            # retrieve and average the horizontal, diagonal and vertical 
            # components
            horizontal=coeff.horizontal.mean(axis=1)
            diagonal=coeff.diagonal.mean(axis=1)
            vertical=coeff.vertical.mean(axis=1)

            # compute the absolute value of the gradients and normalize
            horizontal=np.abs(horizontal)
            vertical=np.abs(vertical)
            diagonal=np.abs(diagonal)

            if self.normalize_coeffs:

                horizontal/=horizontal.max()
                diagonal/=diagonal.max()
                vertical/=vertical.max()

            # add the coefficients 
            visualization[:,start_index:end_index,start_index:end_index]=diagonal
            visualization[:,start_index:end_index,:start_index]=vertical
            visualization[:,:start_index,start_index:end_index]=horizontal
 
        return visualization



def _reproject_wam(coeffs, normalize_coeffs):
    """
    returns the importance of the wavelet
    coefficients on the wavelet transform
    of the image

    coeffs is a list that contains the 
    wavelet coefficients of the image

    the gradients are averaged across the channels

    returns an array (batch_size,img_size,img_size)
    """

    # retrieve the dimensions of the image and 
    # the batch size
    batch_size=coeffs[0].shape[0]
    img_size= int(2 * coeffs[-1].horizontal.shape[-1]) # the image size
                                                        # is twice the size of the 
                                                        # last coefficients

    visualization=np.zeros((
        batch_size, img_size, img_size
    )) # we average the gradient values across channels

    # add the approximation coefficients on the 
    # upper left corner
    approx=coeffs[0].mean(axis=1)
    approx=np.abs(approx)

    if normalize_coeffs:
        approx/=approx.max()

    visualization[:,:approx.shape[1], :approx.shape[2]]=approx #/approx.max()

    # add the approximation coefficients
    for i, coeff in enumerate(coeffs[1:][::-1]):
        
        end_index=int(img_size/2**i)
        start_index=int(img_size/2**(i+1))

        # retrieve and average the horizontal, diagonal and vertical 
        # components
        horizontal=coeff.horizontal.mean(axis=1)
        diagonal=coeff.diagonal.mean(axis=1)
        vertical=coeff.vertical.mean(axis=1)

        # compute the absolute value of the gradients and normalize
        horizontal=np.abs(horizontal)
        vertical=np.abs(vertical)
        diagonal=np.abs(diagonal)

        if normalize_coeffs:

            horizontal/=horizontal.max()
            diagonal/=diagonal.max()
            vertical/=vertical.max()

        # add the coefficients 
        visualization[:,start_index:end_index,start_index:end_index]=diagonal
        visualization[:,start_index:end_index,:start_index]=vertical
        visualization[:,:start_index,start_index:end_index]=horizontal

    return visualization

class WaveletAttribution2D(BaseWAM2D):
    def __init__(self, 
                 model, 
                 wavelet="haar", 
                 method="smooth",
                 J=3, 
                 device=None,
                 mode="reflect",
                 approx_coeffs=False, 
                 normalize_coeffs=True,
                 n_samples=25, 
                 stdev_spread=0.25, # range [.2-.3] produces the best results
                                    # visually 
                 random_seed=42):
            super().__init__(model, 
                             wavelet=wavelet, 
                             J=J, 
                             device=device,
                             mode=mode,
                             approx_coeffs=approx_coeffs,
                             normalize_coeffs=normalize_coeffs)

            """
            self.model=model
            self.wavelet=wave
            self.J=J
            self.approx_coeffs=approx_coeffs
            """        
            self.method=method
            self.n_samples=n_samples
            self.stdev_spread=stdev_spread
            self.random_seed=random_seed

            self.wam=BaseWAM2D(model,wavelet=wavelet,J=J,mode=mode,device=device,approx_coeffs=approx_coeffs,normalize_coeffs=normalize_coeffs)


    def smooth_gradcam(self,x,y):
        """
        implmeents the wam by smoothing the gradients as decribed
        by Smilkov et al (2017)
        """
        # generate the noisy samples
        np.random.seed(self.random_seed)

        # average gradients
        avg_gradients=np.zeros((x.shape[0],x.shape[2],x.shape[3]))

        for _ in range(self.n_samples):

            noisy_x=torch.zeros(x.shape)

            for i in range(x.shape[0]): # iterate over the batch of images

                max_x=x[i,:,:,:].max()
                min_x=x[i,:,:,:].min()

                stdev=self.stdev_spread*(max_x-min_x).cpu()
                # generate noise calibrated for the current image
                noise=np.random.normal(0,stdev,x.shape[1:]).astype(np.float32)
                # apply the noise to the images
                noisy_x[i,:,:,:]=x[i,:,:,:]+torch.tensor(noise)

            # compute the wam
            avg_gradients+=self.wam(noisy_x,y)


        for k in range(avg_gradients.shape[0]): # compute the mean
            avg_gradients[k,:,:]/=self.n_samples

        # reproject the average gradients
        self.scales=self.reproject_wam(avg_gradients,self.J,self.approx_coeffs)

        return avg_gradients

    def intergrated_wam(self,x,y):
        """
        implements the integrated gradients approach for smoothing
        as described by Sundararajan et al (2017)

        In order to approximate from a finite number of steps, the implementation here use the
        trapezoidal rule and not a Riemann sum (see the paper below for a comparison of the results).
        Ref. Computing Linear Restrictions of Neural Networks (2019).
        https://arxiv.org/abs/1908.06214
        """

        # compute the wavelet transform of the input
        # corresponds to z
        coeffs=ptwt.wavedec2(x,self.wavelet, level=self.J, mode=self.mode)

        # convert the coeffs as a numpy array
        arr_coeffs=to_numpy(coeffs, 2, grads=False)
        baseline_z=_reproject_wam(arr_coeffs, normalize_coeffs=True)

        # generate alpha with a given number of steps
        alphas=np.linspace(0,1,self.n_samples)

        # generate the array that accuulates the gradients along
        # the path
        grad_path=np.empty((x.shape[0], self.n_samples, x.shape[2],x.shape[3]), dtype=np.float32)

        for i,alpha in enumerate(alphas):

            # compute the perturbed alpha * z wavelet transform
            path_coeffs= self.alter(alpha,coeffs)
            # evaluate the perturbed wt

            grad_path[:,i,:,:]=self.wam(path_coeffs,y,image=False)

        # once computed the path, average the integral using the 
        # trapezoidal rule
        intergral=np.trapz(np.nan_to_num(grad_path), axis=1)

        # return the results


        self.scales=self.reproject_wam(baseline_z*intergral,self.J,self.approx_coeffs)
        
        return baseline_z*intergral#arr_coeffs*grad_path
    
    def alter(self, alpha, coeffs):
        """
        "multiplies" the coeffs by alpha
        """

        altered_coeffs=[coeffs[0] * alpha]
        for coeff in coeffs[1:]:
            altered_coeffs.append(
                ptwt.constants.WaveletDetailTuple2d(
                    coeff.horizontal * alpha,
                    coeff.vertical * alpha,
                    coeff.diagonal * alpha
                )
            )

        return altered_coeffs


    def __call__(self,x,y):

        if self.method=='smooth':
            return self.smooth_gradcam(x,y)
        
        if self.method=="integratedgrad":
            return self.intergrated_wam(x,y)
    
    
    def reproject_wam(self,average_gradients,num_levels,approx_coeffs=False):
        """
        reprojects the wam in the pixel domain
        
        returns a [n,J,image_size,image_size] array
        where J corresponds to the Jth level
        orientations in a given level are collapsed 
        """

        # retrieve the parameters
        batch_size=average_gradients.shape[0]
        img_size=average_gradients.shape[1]

        if approx_coeffs:
            visualization=np.zeros((batch_size,num_levels+1,img_size,img_size))

        else:
            visualization=np.zeros((batch_size,num_levels,img_size,img_size))
        
        for j in range(self.J):
            # retrieve the vertical, diagonal and horizontal 
            # details at each level
            end_index=int(img_size/2**j)
            start_index=int(img_size/2**(j+1))
            
            diagonal=average_gradients[:,start_index:end_index,start_index:end_index]
            vertical=average_gradients[:,start_index:end_index,:start_index]
            horizontal=average_gradients[:,:start_index,start_index:end_index]
            
            # upsample the coefficients
            for bi in range(batch_size):
                tmp_h=cv2.resize(horizontal[bi,:,:], (img_size,img_size), interpolation=cv2.INTER_LINEAR)
                tmp_v=cv2.resize(vertical[bi,:,:], (img_size,img_size), interpolation=cv2.INTER_LINEAR)
                tmp_d=cv2.resize(diagonal[bi,:,:], (img_size,img_size), interpolation=cv2.INTER_LINEAR)

                visualization[bi,j,:,:]=tmp_h+tmp_v+tmp_d

        # add the approximation coefficients 
        if approx_coeffs:
            for bi in range(batch_size):

                end_index=int(img_size/2**self.J)
                tmp_a=average_gradients[bi,:end_index,:end_index]
                visualization[bi,num_levels,:,:]=cv2.resize(
                    tmp_a, (img_size,img_size), interpolation=cv2.INTER_LINEAR
                )
                

        return visualization
