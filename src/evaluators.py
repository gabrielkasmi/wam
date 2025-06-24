#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')


import numpy as np
from lib.evaluation_helpers import generate_images, generate_masks, generate_images_baseline, generate_subsets,\
    compute_auc,sum_importance, evaluate, reconstruct_images, reconstruct_images_baseline, compute_melspec,\
    SmoothGrad, GradCAM, GradCAMPlusPlus
from lib.wam_1D import WaveletAttribution1D, to_numpy
from lib.wam_2D import WaveletAttribution2D
import ptwt
from lib.helpers import show
import tqdm
import numpy as np
import pywt
import torch
import torchvision.transforms as transforms
from scipy.stats import spearmanr
from scipy.ndimage import zoom, gaussian_filter
from captum.attr import Saliency, IntegratedGradients, GuidedBackprop
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# To implement LRP with EpsilonPlusFLat Canonizer
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer

# Implement SRD
from lib.srd.util import *
from lib.srd.modules.resnet import resnet50

# 1D case 

# class to evaluate the wam 
class Eval1DWAM(WaveletAttribution1D):
    def __init__(self, model, batch_size=128,
                 wavelet="haar", J=3, method="smooth", mode="reflect", device=None, 
                 approx_coeffs=False, n_mels=128, n_fft=1024, sample_rate=44100, n_samples=25, 
                 stdev_spread=0.001, random_seed=42):
        super().__init__(model, wavelet, J, method, mode, device, approx_coeffs, n_mels, n_fft, 
                         sample_rate, n_samples, stdev_spread, random_seed)

        self.grad_wams=None
        self.batch_size=batch_size
        self.gradwam=WaveletAttribution1D(model,wavelet=wavelet,J=J,method=method,mode=mode,device=device,
                                 approx_coeffs=approx_coeffs,n_mels=n_mels,n_fft=n_fft,sample_rate=sample_rate,
                                 n_samples=n_samples,stdev_spread=stdev_spread,random_seed=random_seed)
        
        self.insertion_curve=[]
        self.deletion_curve=[]

    def auc_from_wavelet(self, gradients, waveform, sample, mode, n_iter):
        """
        for one sample, returns a batch of reconstructed soudwaves 
        based on the filtering of their wavelet coefficients

        gradually adds or removes components depending
        on the mode specified (insertion or deletion)

        retruns a batch of melspecs computed from the reconstructions

        if mask is not none, should be a list
        with the transformation for each level ready to be applied to 
        the wavelet transform of the waveform
        """

        coeffs=ptwt.wavedec(waveform, wavelet=self.wavelet, level=self.J)
        coeffs=to_numpy(coeffs,1,grads=False)

        # indices of the decomposition in 1D
        # as we flatten the wavelet decomposition
        scales_indices=[int(waveform.shape[0] / 2**j) for j in range(self.J+1)]
        scales_indices.append(0)
        # retrieve the gradients
        # and concatenate them
        sample_gradients=np.concatenate([g[sample] for g in gradients])


        # create the masks
        insertion_masks=np.zeros((n_iter+1,sample_gradients.shape[0]))
        deletion_masks=np.ones((n_iter+1,sample_gradients.shape[0]))

        # sort the coordinates of the gradients
        # across all scales
        indices=np.argsort(np.abs(sample_gradients))[::-1]
        n_components=int(len(indices)/n_iter)

        # create the masks
        for i in range(n_iter):
            
            corresponding_coords=indices[:min((i+1)*n_components, len(indices))]
            insertion_masks[i+1,corresponding_coords]=1
            deletion_masks[i+1,corresponding_coords]=0

        insertion_masks[-1]=np.ones(indices.shape[0])
        deletion_masks[-1]=np.zeros(indices.shape[0])

        # filter the gradient coefficients
        # and return a stacked array ready for reconstruction
        filtered_coeffs=[]
        for i, coeff in enumerate(coeffs):

            # start and end index for the mask
            start_index=scales_indices[::-1][i]
            end_index=scales_indices[::-1][i+1]

            # filter the coefficients
            if mode=="insertion":

                if insertion_masks[:,start_index:max(end_index,coeff.shape[0])].shape[1] != coeff.shape[0]:
                    mask=np.pad(insertion_masks[:,start_index:max(end_index,coeff.shape[0])], ((0, 0), (0, 1)), mode='constant', constant_values=1)
                
                else:
                    mask=insertion_masks[:,start_index:max(end_index,coeff.shape[0])]

            elif mode=="deletion":
                if deletion_masks[:,start_index:max(end_index,coeff.shape[0])].shape[1] != coeff.shape[0]:
                    mask=np.pad(deletion_masks[:,start_index:max(end_index,coeff.shape[0])], ((0, 0), (0, 1)), mode='constant', constant_values=0)
                
                else:
                    mask=deletion_masks[:,start_index:max(end_index,coeff.shape[0])]

            filtered_coeffs.append(coeff*mask)

        reconstruction=pywt.waverec(filtered_coeffs,wavelet=self.wavelet)

        # Clear temporary variables to avoid memory overload
        del coeffs
        del scales_indices
        del sample_gradients
        del indices
        del filtered_coeffs
        del mask
        torch.cuda.empty_cache()  # Optional: Clear GPU cache if you're using CUDA

        return self.compute_melspec(torch.tensor(np.array([wf/wf.max() for wf in reconstruction]).astype(np.float32)),
                                    self.n_fft,
                                    self.sample_rate,
                                    self.n_mels)
    
    def auc_from_melspec(self, melspec, source_melspec, mode, n_iter):
        """

        for one sample, generates a tensor [n_iter, 1, :, :] 
        of perturbed melspecs, either trough insertion or through deletion

        from the explanation melspec, filters the audio melspec

        if mask is not none: directly applies the mask to the melspec
        

        args:
            melspec (np.array of shape [n_fft, n_mels]) : contains the gradients
            source_melspec (np.array of shape [n_fft, n_mels]) : melspec of the sound
            mode: insertion or deletion
            n_iter : numbre of iterations to approximate the AUC

        returns a Tensor of shape [n_iter, 1, n_fft, n_mels]
        """
        # generate masks from the melspec
        insertion_masks, deletion_masks=generate_masks(n_iter, melspec)

        # remove elements from the source_melspec or 
        if mode=="insertion":
            perturbed_melspecs=insertion_masks * source_melspec

        elif mode=="deletion":
            perturbed_melspecs=deletion_masks * source_melspec

        # convert as tensor and reshape to fit the requirements of the 
        # model
        return torch.tensor(perturbed_melspecs).float().unsqueeze(1)

    def evaluate_auc(self,x,y,mode,target,n_iter=64, argmax=False):
        """
        base to compute the insertion and the deletion scores

        mode: if insertion or deletion
        instance: 'wavelet' or 'melspec': see the delta in predicted prob when
                                          removing components from the melspec or from
                                          the waveform
        """
        # compute the gradwams if missing

        if isinstance(x,list): # convert the input as tensor if necessary
            x=torch.tensor(
                np.array([wf/wf.max() for wf in x]).astype(np.float32)
            )

        # in this case, it is a tuple with the melspecs and the gradients
        if self.grad_wams is None:
            self.grad_wams=self.gradwam(x,y)

        # compute the source melspec of the
        source_melspecs=self.compute_melspec(x, n_fft=self.n_fft, 
                                             sample_rate=self.sample_rate, 
                                             n_mels=self.n_mels)
        
        # convert as an array
        source_melspecs=source_melspecs.squeeze(1).detach().cpu().numpy()
        melspecs, gradients=self.grad_wams
        # retrieve the number of samples
        n_sounds=x.shape[0]

        scores=[]
        predicted_probs=[]
        raw_preds=[]

        for sample in range(n_sounds):

            if target=="melspec":
                perturbed_samples=self.auc_from_melspec(melspecs[sample,:,:], source_melspecs[sample,:,:], mode, n_iter)

            elif target=="wavelet":
                perturbed_samples=self.auc_from_wavelet(gradients, x[sample], sample, mode, n_iter)
            
            preds=self.model(perturbed_samples.to(self.device)).cpu().detach().numpy()

            raw_preds.append(preds)

            probs=np.exp(preds)/np.sum(np.exp(preds), axis=1, keepdims=True)
            predicted_prob=probs[:,y[sample]]

            scores.append(compute_auc(predicted_prob))     
            predicted_probs.append(predicted_prob)
            
        del perturbed_samples, preds, probs, predicted_prob
        if argmax:
            return raw_preds
        else:
            return scores, predicted_probs
        
    def insertion(self, x,y,target,n_iter):
        scores, predicted_probs=self.evaluate_auc(x,y,'insertion',target,n_iter)
        self.insertion_curves=predicted_probs
        return scores
    
    def deletion(self, x,y,target,n_iter):
        scores, predicted_probs=self.evaluate_auc(x,y,'deletion',target,n_iter)
        self.deletion_curves=predicted_probs
        return scores
    
    def faithfulness_of_spectra(self,x,y,target):
        """
        evaluates the gradwam according to the faithfulness of spectra (Parekh et al) 
        
        computes the drop in predicted probability for the class of interest after the input 
        spectrum is filtered out 

        FF_i = f(x_i)_c - f(x_i * mask)_c

        where mask(h) is a mask computed with the explanation

        the larger the better, as it indicates that the explanation found relevant part of the spectra

        Values can be negative, indicating a poor faithfulness. The final score corresponds to the median of 
        the FF.

        *
        in this implementation, rely on our implementation of the deletion and n_iter=2. The Oth mask being the 
        unaltered image x_i, the last one is completely masked out and the middle one as a mask that covers half 
        of the coefficients, ranked by order of importance (x_i * mask)
        *
        """

        # retrieve the predicted probs 
        _, predicted_probs=self.evaluate_auc(x,y,'deletion',target,2)

        # convert as an array
        predicted_probs=np.array(predicted_probs)


        return (predicted_probs[:,0] - predicted_probs[:,1]).tolist()

    def input_fidelity(self, x, y, target):
        """
        evaluates with the input fidelity (Paissan et al, 2023) 

        which is the indicator that the predicted class 
        without mask is the same as the predicted class with the mask-only 
        portion of the input signal.

        Values are then averaged

        *
        our implementation is based on the insertion, but rather than predicted 
        probabilities, we return the argmax
        *
        """

        preds=self.evaluate_auc(x,y,'insertion',target,2, argmax=True)

        preds=np.array(preds)
        preds = preds[:, 1:, :]  # Remove axis 1 at index 0, this avoids copying data unnecessarily
        # axis 1, index 0 indicates a completely empty signal (no features inserted)
        # axis 1, index 1 has half of the features
        # axis 1, index 3 has all features 

        # Vectorized argmax computation for the remaining axes
        argmaxes = np.argmax(preds, axis=2)  # Shape will now be (n_samples, 2)

        return argmaxes.tolist()

# class to evaluat the baseline

class EvalAudioBaselines():
    def __init__(self,
                 method_name,
                 model,
                 device=None,
                 n_fft=1024,
                 sample_rate=44100,
                 n_mels=128,
                 batch_size=64,
                 ):
        
        self.n_fft=n_fft
        self.n_mels=n_mels
        self.sample_rate=sample_rate
        self.batch_size=batch_size

        if device is not None:
            model=model.to(device)
            self.model=model
            self.device=device
        
        else:
            device=next(model.parameters()).device
            self.model=model
            self.device=device
        
        self.method_name=method_name
        self.explanations=None

        if method_name=="integratedgrad":
            self.method=IntegratedGradients(model)

        elif method_name=="gradcam":
            layer=model.layer
            self.method=GradCAM(model,layer)

            
        elif method_name=="smoothgrad":
            self.method=SmoothGrad(model, stdev_spread=0.001) # same spread as the wam

        elif method_name=="saliency":
            self.method=Saliency(model)
            
    def compute_explanations(self, x, y):
        """
        computes the explanations, returns the explanation
        on the melspec of the input
        """

        # compute the wavelet transform of the sound wave
        if isinstance(x, list):
            x=torch.tensor(np.array([wf/wf.max() for wf in x]).astype(np.float32))

        # convert x to the correct shape if not already the case
        # compute the melspecs
        input_melspecs=compute_melspec(x,self.n_fft,
                                       self.sample_rate,
                                       self.n_mels).to(self.device)

        if self.method_name=="integratedgrad":

            explanations=self.method.attribute(input_melspecs, 
                                           target=torch.tensor(y).to(self.device),
                                           method="riemann_trapezoid",
                                           return_convergence_delta=False,
                                           n_steps=50,
                                           internal_batch_size=self.batch_size
                                           ).squeeze(1).detach().cpu().numpy()
            
        elif self.method_name=="smoothgrad":

            explanations=self.method(input_melspecs,y)

        elif self.method_name=="gradcam":

            explanations=self.method(input_melspecs,y)

        elif self.method_name=="saliency":
            explanations=self.method.attribute(input_melspecs,
                                               target=torch.tensor(y).to(self.device)).squeeze(1).detach().cpu().numpy()


        return explanations

    def auc_from_melspec(self, melspec, source_melspec, mode, n_iter):
        """

        for one sample, generates a tensor [n_iter, 1, :, :] 
        of perturbed melspecs, either trough insertion or through deletion

        from the explanation melspec, filters the audio melspec

        if mask is not none: directly applies the mask to the melspec
        

        args:
            melspec (np.array of shape [n_fft, n_mels]) : contains the gradients
            source_melspec (np.array of shape [n_fft, n_mels]) : melspec of the sound
            mode: insertion or deletion
            n_iter : numbre of iterations to approximate the AUC

        returns a Tensor of shape [n_iter, 1, n_fft, n_mels]
        """
        # generate masks from the melspec
        insertion_masks, deletion_masks=generate_masks(n_iter, melspec)

        # remove elements from the source_melspec or 
        if mode=="insertion":
            perturbed_melspecs=insertion_masks * source_melspec

        elif mode=="deletion":
            perturbed_melspecs=deletion_masks * source_melspec

        # convert as tensor and reshape to fit the requirements of the 
        # model
        return torch.tensor(perturbed_melspecs).float().unsqueeze(1)
        
    def evaluate_auc(self, x, y, mode, n_iter=64, argmax=False):
        """
        base to compute the insertion and the deletion scores

        mode: if insertion or deletion
        instance: 'wavelet' or 'melspec': see the delta in predicted prob when
                                          removing components from the melspec or from
                                          the waveform
        """

        if isinstance(x, list):
            x=torch.tensor(np.array([wf/wf.max() for wf in x]).astype(np.float32))

        # compute the compute the explanations
        melspecs=self.compute_explanations(x,y)

        # compute the source melspec of the
        source_melspecs=compute_melspec(x, n_fft=self.n_fft, 
                                             sample_rate=self.sample_rate, 
                                             n_mels=self.n_mels)
        
        # convert as an array
        source_melspecs=source_melspecs.squeeze(1).detach().cpu().numpy()
        # retrieve the number of samples
        n_sounds=melspecs.shape[0]

        scores=[]
        predicted_probs=[]
        raw_preds=[]

        for sample in range(n_sounds):

            perturbed_samples=self.auc_from_melspec(melspecs[sample,:,:], source_melspecs[sample,:,:], mode, n_iter)
            
            preds=self.model(perturbed_samples.to(self.device)).cpu().detach().numpy()

            raw_preds.append(preds)

            probs=np.exp(preds)/np.sum(np.exp(preds), axis=1, keepdims=True)
            predicted_prob=probs[:,y[sample]]

            scores.append(compute_auc(predicted_prob))     
            predicted_probs.append(predicted_prob)
            
        del perturbed_samples, preds, probs, predicted_prob
        if argmax:
            return raw_preds
        else:
            return scores, predicted_probs

    def insertion(self,x,y, n_iter):
        scores, predicted_probs=self.evaluate_auc(x,y,'insertion',n_iter)
        self.insertion_curves=predicted_probs
        return scores

    def deletion(self, x,y, n_iter):
        """
        computes the deletion
        """
        scores, predicted_probs=self.evaluate_auc(x,y,'deletion',n_iter)
        self.insertion_curves=predicted_probs
        return scores

    def faithfulness_of_spectra(self,x,y):
        """
        evaluates the gradwam according to the faithfulness of spectra (Parekh et al) 
        
        computes the drop in predicted probability for the class of interest after the input 
        spectrum is filtered out 

        FF_i = f(x_i)_c - f(x_i * mask)_c

        where mask(h) is a mask computed with the explanation

        the larger the better, as it indicates that the explanation found relevant part of the spectra

        Values can be negative, indicating a poor faithfulness. The final score corresponds to the median of 
        the FF.

        *
        in this implementation, rely on our implementation of the deletion and n_iter=2. The Oth mask being the 
        unaltered image x_i, the last one is completely masked out and the middle one as a mask that covers half 
        of the coefficients, ranked by order of importance (x_i * mask)
        *
        """

        # retrieve the predicted probs 
        _, predicted_probs=self.evaluate_auc(x,y,'deletion',2)

        # convert as an array
        predicted_probs=np.array(predicted_probs)

        return (predicted_probs[:,0] - predicted_probs[:,1]).tolist()

    def input_fidelity(self, x,y):
        """
        evaluates with the input fidelity (Paissan et al, 2023) 

        which is the indicator that the predicted class 
        without mask is the same as the predicted class with the mask-only 
        portion of the input signal.

        Values are then averaged

        *
        our implementation is based on the insertion, but rather than predicted 
        probabilities, we return the argmax
        *
        """

        preds=self.evaluate_auc(x,y,'insertion',2, argmax=True)

        preds=np.array(preds)
        preds = preds[:, 1:, :]  # Remove axis 1 at index 0, this avoids copying data unnecessarily
        # axis 1, index 0 indicates a completely empty signal (no features inserted)
        # axis 1, index 1 has half of the features
        # axis 1, index 3 has all features 

        # Vectorized argmax computation for the remaining axes
        argmaxes = np.argmax(preds, axis=2)  # Shape will now be (n_samples, 2)

        return argmaxes.tolist()
# 2D case 
# evaluation of the wam

# this class evaluates the Gradwam
class Eval2DWAM(WaveletAttribution2D):
    """
    wrapper that evaluates the gradwam 2D.

    Features the following metrics: 
    Insertion, Deletion (Petsiuk et al)
    mu-Fidelity (Bhatt et al)
    """

    def __init__(self,
                 model, 
                 wavelet="haar", 
                 J=3, 
                 device=None,
                 mode="reflect",
                 transform=None,
                 approx_coeffs=False, 
                 method="smooth",
                 n_samples=25, 
                 batch_size=128,
                 stdev_spread=0.25, # range [.2-.3] produces the best results
                                    # visually 
                 random_seed=42):
            super().__init__(model, 
                             wavelet=wavelet, 
                             J=J, 
                             device=device,
                             mode=mode,
                             approx_coeffs=approx_coeffs)
            
            self.smooted_grad_wam=WaveletAttribution2D(model,wavelet=wavelet,J=J,mode=mode,
                                              approx_coeffs=approx_coeffs, 
                                              n_samples=n_samples,
                                              stdev_spread=stdev_spread,
                                              random_seed=random_seed,
                                              method=method,
                                              device=device)
            self.batch_size=batch_size
            self.grad_wams=None

            if transform is None:
                self.transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            
            # debug attributes
            self.insertion_curves=[]
            self.deletion_curves=[]

    def evaluate_auc(self,x,y,mode, n_iter=64):
        """
        computes the insertion score
        """
            
        # compute the wam associated with the images
        if self.grad_wams is None:
            self.grad_wams=self.smooted_grad_wam(x,y)

        # retrieve the number of images
        n_samples=self.grad_wams.shape[0]
        images=[
              show(x[i,:,:,:], plot=False) for i in range(x.shape[0])
        ]

        scores=[]
        predicted_probs=[]

        for sample in range(n_samples):            
            # retrieve the explanations and the images
            wam=self.grad_wams[sample,:,:]

            # generate the altered images  
            altered_images=generate_images(images[sample],wam, self.J, n_iter, mode, wavelet=self.wavelet)

            # evaluate the model on the altered images
            x_t=torch.stack([
                 self.transform(im).float().to(self.device) for im in altered_images
            ]).to(self.device)

            #with torch.no_grad():
            preds=self.model(x_t).cpu().detach().numpy()
            probs=np.exp(preds)/np.sum(np.exp(preds), axis=1, keepdims=True)

            # retrieve the probs for the yi_th column, which corresponds
            # to the grund truth label
            predicted_prob=probs[:,y[sample]]

            # compute the auc       
            scores.append(compute_auc(predicted_prob))     
            predicted_probs.append(predicted_prob)

        return scores, predicted_probs
    
    def insertion(self,x,y,n_iter=64):
        """
        computes the insertion
        """
        scores, predicted_probs=self.evaluate_auc(x,y,"insertion", n_iter=n_iter)
        self.insertion_curves=predicted_probs
        return scores

    def deletion(self,x,y,n_iter=64):
        """
        computes the deletion
        """

        scores,predicted_probs=self.evaluate_auc(x,y,"deletion", n_iter=n_iter)
        self.deletion_curves=predicted_probs

        return scores
    
    def mu_fidelity(self,x,y,
                    grid_size=28,
                    sample_size=128,
                    subset_size=157):
        """
        computes the mu fidelity as:

        mean(spearmanr(attr, base - alteration))

        where attr is the sum of importance of the features in I
        base = f(x) is the prediction without alteration for the image
        alteration is the prediction where the features in I are set in the baseline state

        the set K is equal to 20% of the variables and the baseline is x=0
        (as done in Novello et al, 2022)

        the feature set is taken over superpixels to ease the computations. The resuling masks 
        are upsampled to the size of the input image

        """
        # compute the explanations

        if self.grad_wams is None:
            self.grad_wams=self.smooted_grad_wam(x,y)

        # seed for the reproducibility
        np.random.seed(self.random_seed)

        # compute the baseline probabilities 
        base_preds = self.model(x.to(self.device)).cpu().detach().numpy()

        # null_image=torch.zeros(x.shape).to(self.device)
        # base_preds = model(null_image).cpu().detach().numpy()
        # fin de la modification

        base_probs=np.exp(base_preds)/np.sum(np.exp(base_preds), axis=1, keepdims=True)
        base_probs=[base_probs[i,y[i]] for i in range(len(y))] # retrieve the predicted probabilities at the corresponding indices

        images=[
              show(x[i,:,:,:], plot=False) for i in range(x.shape[0])
        ]
        mu_fidelities=[]

        # clear the cache
        torch.cuda.empty_cache()

        for i in tqdm.tqdm(range(len(self.grad_wams))):

            wam=gaussian_filter(self.grad_wams[i], sigma=2)         

            indices=generate_subsets(grid_size,subset_size,sample_size)
            # indices corresponds to the coordinates of the superpixels

            # generate the baseline mask
            baseline_mask=self.compute_baseline_state(images[i],y[i],grid_size,self.batch_size,sample_size)

            # generate the altered images (corresponds to f(x_{x_u = \bar{x}}))
            masks=np.ones((sample_size,grid_size,grid_size))

            for j,index_set in enumerate(indices):
                cx,cy=zip(*index_set)
                masks[j,cx,cy]=baseline_mask[cx,cy]

            # upsample the masks and generate the perturbed images
            zoom_factor=(1,x.shape[2]/grid_size,x.shape[3]/grid_size)
            upsampled_masks=zoom(masks,zoom_factor,order=0) # corresponds to linear interpolation


            altered_images=reconstruct_images(images[i],self.J,upsampled_masks,wavelet=self.wavelet)

            altered_preds=[]

            nb_batch=int(np.ceil(len(altered_images)) / self.batch_size)
            for batch_index in range(nb_batch):

                start_index=batch_index*self.batch_size
                end_index=min(len(altered_images),(batch_index+1)*self.batch_size)

                x_t=torch.stack([
                    self.transform(im).float().to(self.device) for im in altered_images[start_index:end_index]
                ]) 

                altered_preds.append(evaluate(x_t,y[i],self.model,self.batch_size).tolist())
            altered_preds=np.array(
                list(sum(altered_preds, []))
            )

            # compute the preds, i.e.  f(x) - f(x_{x_u = \bar{x}})
            preds=base_probs[i]-altered_preds

            # compute the importance of the features in the subset
            # corresponds to sum(g(x)_i)
            attrs = sum_importance(wam, indices,grid_size,sample_size, batch_size=self.batch_size)

            corrs=spearmanr(preds,attrs)
            mu_fidelity=np.nanmean(corrs)
            mu_fidelities.append(mu_fidelity)

        return mu_fidelities

    def compute_baseline_state(self,image,label,grid_size,batch_size,sample_size):
        """
        computes a baseline state and returns the corresponding (low resolution) 
        mask
        """

        # generate the masks
        source_masks=np.random.uniform(size=(sample_size,grid_size,grid_size))
        
        # upsample the masks
        zoom_factor=(1,image.shape[0]/grid_size,image.shape[1]/grid_size)
        masks=zoom(source_masks,zoom_factor,order=0)

        altered_images=reconstruct_images(image,self.J,masks,wavelet=self.wavelet)
        y=[]

        nb_batch=int(np.ceil(len(altered_images)) / batch_size)
        for batch_index in range(nb_batch):

            start_index=batch_index*batch_size
            end_index=min(len(altered_images),(batch_index+1)*batch_size)

            x=torch.stack([
                self.transform(im).float().to(self.device) for im in altered_images[start_index:end_index]
            ])

            y.append(evaluate(x,label,self.model,batch_size).tolist())

        y=np.array(list(sum(y, [])))

        # get the preds \approx 0 (for the label)
        # corresponds to the argmin
        index=np.argmin(y)
                 
        return source_masks[index,:,:]
    
    # this class evaluates the alternative 

class EvalImageBaselines():
    def __init__(self,
                 method_name,
                 model,
                 layers=None,
                 transform=None,
                 random_seed=42,
                 batch_size=2
                 ):
        
        """
        evaluates baseline gradient based feature attribution 
        methods using the insertion, deletion and mu fidelity metrics

        implementation of the evaluation metrics follows closely the 
        implementation for the gradwam.

        methods are based on two sources: 
            - captum for saliency and the integrated gradients
            (see https://captum.ai/)
            - grad-cam for the gradcam, gradcam++ 
            (see https://jacobgil.github.io/pytorch-gradcam-book/introduction.html)

        pass as input the name of the method among the following:
        {"gradcam", "gradcampp", "intergratedgrad", "saliency"} 
        and the implementation is automatically carried out

        """

        if transform is None:
            self.transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform=transform

        self.random_seed=random_seed
        self.method_name=method_name
        self.batch_size=batch_size
        self.model=model
        self.explanations=None
        self.device=next(model.parameters()).device


        if method_name=="gradcam":
            # implement the gradcam from grad-cam 
            # requires the layers to be of the form 
            # layers=[model.layer4[-1]] for the resnet 

            self.method=GradCAM(model,layers)#,reshape_transform=reshape_transform)

        elif method_name=="gradcampp":
            # implement the gradcampp from grad-cam 
            # requires the layers to be of the form 
            # layers=[model.layer4[-1]] for the resnet 

            self.method=GradCAMPlusPlus(model,layers)#,reshape_transform=reshape_transform)

        elif method_name=="saliency":
            # implement from captum
            self.method=Saliency(model)

        elif method_name=="integratedgrad":
            # implement from captum
            # with a lighter computational burden
            self.method=IntegratedGradients(model)

        elif method_name=="smoothgrad":
            # custom implementation of the smoothgrad
            self.method=SmoothGrad(model)

#        # baslines added following the rebuttal
#        elif method_name=="attnlrp":
#            #self.method=#TODO https://github.com/rachtibat/LRP-eXplains-Transformers # will only work for the ViT (+ ConvNext ?) backbones
#
        elif method_name=="guided_backprop":
            self.method=GuidedBackprop(model)
#        
        elif method_name=="layercam":
            self.method=LayerCAM(model,[layers]) # LayerCAM to implement https://github.com/jacobgil/pytorch-grad-cam
#
        elif method_name=="lrp":

            canonizer=ResNetCanonizer() # manually change the canonize for the other mo
                                        # models ? 
            composite=EpsilonPlusFlat(canonizers=[canonizer])
            self.method=Gradient(model=model, composite=composite)
 
        elif method_name=="srd":
            # only works for ResNet50 with a custom implementation
            model_org = resnet50(pretrained=True,name='resnet18').to(self.device)
            model_org = model_org.eval()
            infer_pkg = InferencePkg()
            infer_pkg.device = self.device
            self.infer_pkg=infer_pkg
            self.method=model_org

    def compute_explanations(self,x,y):
        """
        computes the explanations given the inputs 
        the labels y may differ in specificatoins depending on 
        the method used to compute the explanations 
        """

        x=x.to(self.device)

        if self.method_name=="srd":
            print('coucou')
            _=self.method(x,self.infer_pkg)
            y=torch.eye(1000)[y].to(self.device)
            simmap = self.method.get_pixel_cont(x,y.long(),self.infer_pkg,lambda x:x).squeeze().detach().cpu()
            explanations=simmap.detach().cpu().numpy()
            return explanations

        if self.method_name=="layercam":
            return self.method(x,[ClassifierOutputTarget(yy) for yy in y])

        if self.method_name == 'smoothgrad':

            return self.method(x,y)
        
        elif self.method_name in ['gradcam', "gradcampp"]:

            # x.requires_grad
            return self.method(x,y)
        else: 

            # check that everybody is on the same device
            # some rework is also needed to generate an array from the 
            # output tensor

            if not isinstance(y, torch.Tensor):
                y=torch.tensor(y).to(self.device)


            if self.method_name=="integratedgrad":
                explanations=self.method.attribute(x,target=y,
                                            method="riemann_trapezoid",
                                            return_convergence_delta=False,
                                            n_steps=50,
                                            internal_batch_size=self.batch_size
                                            )
            
            if self.method_name=="lrp":
                targets=torch.eye(1000)[y.detach().cpu()].to(self.device)
                _, attribution=self.method(x,targets)

                return np.mean(attribution.cpu().detach().numpy(), axis=1)
            else:
                explanations=self.method.attribute(x,target=y)

            explanations=np.transpose(explanations.cpu().detach().numpy(), (0, 2, 3, 1))
            return np.mean(explanations, axis=-1)      
        
    def evaluate_auc(self,x,y,mode,n_iter=64):
        """
        computes the insertion score
        """

        # retrieve the number of images
        n_samples=x.shape[0]
        
        images=[
              show(x[i,:,:,:].detach(), plot=False) for i in range(x.shape[0])
        ]

        # compute the explanations
        if self.explanations is None:
            self.explanations=self.compute_explanations(x,y)
            torch.cuda.empty_cache()

        scores=[]
        predicted_probs=[]

        for sample in range(n_samples):            
            # retrieve the explanations and the images
            attr=self.explanations[sample,:,:]

            # generate the altered images  
            altered_images=generate_images_baseline(images[sample],attr, n_iter, mode)

            # evaluate the model on the altered images
            x_t=torch.stack([
                 self.transform(im).float().to(self.device) for im in altered_images
            ])

            nb_batch=int(np.ceil(x_t.shape[0] / self.batch_size))
            predicted_prob=[]

            for batch_index in range(nb_batch):

                start_index=batch_index*self.batch_size
                end_index=min(x_t.shape[0], (batch_index + 1)*self.batch_size)

                preds=self.model(x_t[start_index:end_index].to(self.device)).cpu().detach().numpy()
                probs=np.exp(preds)/np.sum(np.exp(preds), axis=1, keepdims=True)
                pred_prob=probs[:,y[sample]]
                predicted_prob.append(pred_prob.tolist())

            predicted_prob=list(sum(predicted_prob, []))

            # compute the auc       
            scores.append(compute_auc(np.array(predicted_prob)))     
            predicted_probs.append(predicted_prob)

            # clear the cache
            altered_images.clear()
            torch.cuda.empty_cache()

        return scores, predicted_probs

    def insertion(self,x,y,n_iter=128):
        """
        computes the insertion
        """
        scores, predicted_probs=self.evaluate_auc(x,y,"insertion", n_iter=n_iter)
        self.insertion_curves=predicted_probs
        return scores

    def deletion(self,x,y,n_iter=128):
        """
        computes the deletion
        """

        scores,predicted_probs=self.evaluate_auc(x,y,"deletion", n_iter=n_iter)
        self.deletion_curves=predicted_probs

        return scores
    
    def compute_baseline_state(self, image, label, grid_size, batch_size, sample_size):
        """
        Computes a baseline state and returns the corresponding (low resolution) mask.
        """

        # Generate the masks with the correct dtype
        source_masks = np.random.uniform(size=(sample_size, grid_size, grid_size)).astype(np.float32)
        
        # Upsample the masks
        zoom_factor = (1, image.shape[0] / grid_size, image.shape[1] / grid_size)

        y = []
        nb_batch = int(np.ceil(sample_size / batch_size))

        for batch_index in range(nb_batch):
            start_index = batch_index * batch_size
            end_index = min(sample_size, (batch_index + 1) * batch_size)

            # Upsample masks and ensure they have the correct dtype
            masks = zoom(source_masks[start_index:end_index], zoom_factor, order=0).astype(np.float32)
            altered_images = reconstruct_images_baseline(image, masks)
            x = torch.stack([
                self.transform(im).float().to(self.device) for im in altered_images
            ])

            y.append(evaluate(x, label, self.model, batch_size).tolist())

            torch.cuda.empty_cache()
            altered_images.clear()

        y = np.array(list(sum(y, [])))

        # Get the index of the minimum value in y
        index = np.argmin(y)
        del masks
                 
        return source_masks[index,:,:]

    def mu_fidelity(self,x,y,
                    grid_size=28,
                    sample_size=512,
                    subset_size=157):
        """
        computes the mu fidelity as:

        mean(spearmanr(attr, base - alteration))

        where attr is the sum of importance of the features in I
        base = f(x) is the prediction without alteration for the image
        alteration is the prediction where the features in I are set in the baseline state

        the set K is equal to 20% of the variables and the baseline is x=0
        (as done in Novello et al, 2022)

        the feature set is taken over superpixels to ease the computations. The resuling masks 
        are upsampled to the size of the input image

        """
        # compute the explanations

        if self.explanations is None:
            self.explanations=self.compute_explanations(x,y)

        # seed for the reproducibility
        np.random.seed(self.random_seed)

        # compute the baseline probabilities 
        torch.cuda.empty_cache()
        base_preds = self.model(x.to(self.device)).cpu().detach().numpy()

        # null_image=torch.zeros(x.shape).to(self.device)
        # base_preds = model(null_image).cpu().detach().numpy()
        # fin de la modification

        base_probs=np.exp(base_preds)/np.sum(np.exp(base_preds), axis=1, keepdims=True)
        base_probs=[base_probs[i,y[i]] for i in range(len(y))] # retrieve the predicted probabilities at the corresponding indices

        images=[
              show(x[i,:,:,:], plot=False) for i in range(x.shape[0])
        ]
        mu_fidelities=[]

        # clear the cache
        torch.cuda.empty_cache()

        for i in tqdm.tqdm(range(len(self.explanations))):

            explanation=self.explanations[i]
            indices=generate_subsets(grid_size,subset_size,sample_size)
            # indices corresponds to the coordinates of the superpixels

            # generate the baseline mask
            baseline_mask=self.compute_baseline_state(images[i],y[i],grid_size,self.batch_size,sample_size)

            # generate the altered images (corresponds to f(x_{x_u = \bar{x}}))
            masks=np.ones((sample_size,grid_size,grid_size))

            for j,index_set in enumerate(indices):
                cx,cy=zip(*index_set)
                masks[j,cx,cy]=baseline_mask[cx,cy]

            # upsample the masks and generate the perturbed images
            zoom_factor=(1,x.shape[2]/grid_size,x.shape[3]/grid_size)

            altered_preds=[]

            nb_batch=int(np.ceil(sample_size) / self.batch_size)

            for batch_index in range(nb_batch):

                start_index=batch_index*self.batch_size
                end_index=min(sample_size,(batch_index+1)*self.batch_size)

                upsampled_masks=zoom(masks[start_index:end_index],zoom_factor,order=0) # corresponds to linear interpolation

                altered_images=reconstruct_images_baseline(images[i],upsampled_masks)


                x_t=torch.stack([
                    self.transform(im).float().to(self.device) for im in altered_images
                ])

                altered_preds.append(evaluate(x_t,y[i],self.model,self.batch_size).tolist())

                altered_images.clear()
                del upsampled_masks
                torch.cuda.empty_cache()

            altered_preds=np.array(
                list(sum(altered_preds, []))
            )

            # compute the preds, i.e.  f(x) - f(x_{x_u = \bar{x}})
            preds=base_probs[i]-altered_preds
            # compute the importance of the features in the subset
            # corresponds to sum(g(x)_i)
            attrs = sum_importance(explanation, indices,grid_size,sample_size,batch_size=self.batch_size)

            corrs=spearmanr(preds,attrs)
            mu_fidelity=np.nanmean(corrs)
            mu_fidelities.append(mu_fidelity)

            altered_images.clear()

        return mu_fidelities
    

    