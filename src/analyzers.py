# -*- coding: utf-8 -*-

# Libraries
import sys
sys.path.append('../lib')

import torch
import numpy as np
import torchvision.transforms as transforms

from lib.wam_2D import WaveletAttribution2D
from lib.helpers import show
from lib.analyzers_helpers import generate_partial_image, generate_disentangled_images
from PIL import Image

class WAMAnalyzer2D(WaveletAttribution2D):
    """
    a wrapper to analyze and decompose the images
    following an analysis with the Gradwam

    methods:

    - isolate_scales: returns a decomposition of the image
                      into the important scale components.
                      by summing them, we recover what is important
                      on the image
    - isolate_necessary_components: identifies the necessary information
                                    on the image by evaluating the model
                                    on an image with increasingly more 
                                    components. Similarities with a deletion 
                                    or deletion
    """

    def __init__(self, 
                 model, 
                 wavelet="haar", 
                 J=3, 
                 device=None,
                 mode="reflect",
                 approx_coeffs=False, 
                 method="smooth",
                 n_samples=25, 
                 stdev_spread=0.25, # range [.2-.3] produces the best results
                                    # visually 
                 random_seed=42):
            super().__init__(model, 
                             wavelet=wavelet, 
                             J=J, 
                             device=device,
                             method="smooth",
                             mode=mode,
                             approx_coeffs=approx_coeffs)

            """
            self.model=model
            self.wavelet=wave
            self.J=J
            self.approx_coeffs=approx_coeffs
            """        
            self.n_samples=n_samples
            self.stdev_spread=stdev_spread
            self.random_seed=random_seed

            self.smooted_grad_wam=WaveletAttribution2D(model,wavelet=wavelet,J=J,method=method,mode=mode,approx_coeffs=approx_coeffs, device=device)
            
            self.grad_wams=None
            
            self.insertion_quantile=[]
            self.deletion_quantile=[]
            # self.predicted_probabilities=[]
            

    def isolate_scales(self,x,y, EPS=0.1):
         """
         computes the Gradwam for the given batch of images
         
         returns a list of np.ndarray of shape (J+1,3,W,H)
         """

        # compute the smoothed grad cam
         # update the attribute
         if self.grad_wams is None:
              self.grad_wams=self.smooted_grad_wam(x,y)

         # images: convert the tensor x as a list of np.ndarrays
         images=[
              show(x[i,:,:,:], plot=False) for i in range(x.shape[0])
         ]

         return [generate_disentangled_images(self.grad_wams[i],image, self.J,EPS=EPS,wavelet=self.wavelet)
                 for i, image in zip(range(self.grad_wams.shape[0]), images)
                 ]
    
    def isolate_necessary_components(self,x,y,qs,transform=None,mode=None):
        """
        generates an image by adding or removing components, depending on whether the user specified
        insertion or deletion mode.

        inputs:
        x, y the images and their corresponding class

        transform (opt): the transforms to be applied to the images. If None, passes the 
                         default ImageNet augmentations
        
        mode (opt): insertion or deletion. Either gradually adds components (insertion) 
                                           or removes them (deletion)

        adds a 
        returns a (None, None, grad_wam) if we are unable to recover a correct prediction.

        returns a list(tuple) where each tuple contains the image, the wam 
                and the filtered wam for all images in the batch 

        
        """
        if mode is None:
            print("Input a mode, either 'insertion' or 'deletion'")
            raise ValueError

        if transform is None:
            transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
        # check that for insertion, quantiles are in decreasing order
        # and that for deletion, they are in increasing order

        if mode=="deletion":
            assert qs[0] <= qs[1]
        if mode=="insertion":
            assert qs[0] >= qs[1]

        # compute the smoothed grad cam
        # update the attribute
        if self.grad_wams is None:
             self.grad_wams=self.smooted_grad_wam(x,y)
              
        # for each image, compute and evaluate the prediction of the model

        outs=[]
        for index in range(x.shape[0]):
            # retrieve the image and the label
            xi=x[index,:,:,:]
            yi=y[index]
            grad_wam=self.grad_wams[index]

            img=show(xi,plot=False)
            
            images, masks=[],[]

            for q in qs:
                image, mask=generate_partial_image(img,grad_wam,q,self.J, wavelet=self.wavelet)
                image=((image-image.min()) / image.max()-image.min()) * 255
                images.append(Image.fromarray(image.astype(np.uint8)))
                masks.append(mask)

            # evaluate the model on these images
            x_t=torch.stack([
                transform(im) for im in images
            ]).to(self.device)

            preds=self.model(x_t).cpu().detach().numpy()
            probs=np.exp(preds)/np.sum(np.exp(preds), axis=1, keepdims=True)

            predicted_label=np.argmax(preds,axis=1)

            # retrieve the probs for the yi_th column, which corresponds
            # to the grund truth label
            # predicted_prob=probs[:,yi]
            # self.predicted_probabilities.append(predicted_prob)

            prediction_status=np.where(predicted_label==yi)[0]

            if len(prediction_status):
                if mode=="deletion": #take the first correctly predicted image
                    correct_index=np.where(predicted_label==yi)[0][-1]
                    #print(correct_index)
                    self.deletion_quantile.append(qs[correct_index])

                elif mode=="insertion": # take the last correctly predicted image
                    correct_index=np.where(predicted_label==yi)[0][0]
                    self.insertion_quantile.append(qs[correct_index])
                    #print(correct_index)

                # add the corresponding image, the corresponding components
                # and the overall wam of the image
                outs.append(
                    ((images[0], images[correct_index], images[-1]), masks[correct_index], grad_wam, (probs, correct_index))
                )
            else:
                outs.append(
                    ((None, None, None), None, grad_wam, (None, np.nan))
                )
                #print("Couldn't find the ground truth for this image. Check the ground truth label.")
                continue

                # outs.append((None,None,grad_wam))
                # self.insertion_quantile.append(None)
                # self.deletion_quantile.append(None)

        return outs

