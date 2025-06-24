# -*- coding: utf-8 -*-


# libaries
import torch
import pywt

from PIL import Image
from scipy.ndimage import zoom
import random
import numpy as np
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# custom for the gradcampp

import torch
import torch.nn.functional as F
import numpy as np


def compute_melspec(waveforms, n_fft=1024,sample_rate=44100,n_mels=128):
    """
    compute the melspec of an input batch of tensors
    reconstructed from the wavelet transform with grads

    inputs:
    power: if we keep the power scale or decibels
    reconstruction: the reconstruced soundwaves of shape [N,W]
    sample_rate, n_fft, n_mels: the parameters for the MelSpect
    
    returns a [N,W,H] tensor
    """
    if isinstance(waveforms, list):
        waveforms=torch.tensor(np.array([wf/wf.max() for wf in waveforms]).astype(np.float32))
    
    # instantiate the Melspect and the power_to_db conversion
    power_to_db=AmplitudeToDB()
    mel_spectrogram=MelSpectrogram(sample_rate=sample_rate,n_fft=n_fft,n_mels=n_mels)
    
    melspecs=[]

    for waveform in waveforms:
            
            melspec=mel_spectrogram(waveform)
            melspecs.append(
                power_to_db(melspec).T.squeeze(-1).unsqueeze(0)
            )
    
    return torch.stack(melspecs)

class SaveFeatures:
    def __init__(self, module):
        self.module = module
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
        self.gradients = None
        
    def hook_fn(self, module, input, output):
        self.features = output
        # Register a hook to capture the gradients
        self.grad_hook = output.register_hook(self.save_grads)
        
    def save_grads(self, grad):
        self.gradients = grad
        
    def close(self):
        self.hook.remove()
        if hasattr(self, 'grad_hook'):
            self.grad_hook.remove()

class GradCAMPlusPlus:
    def __init__(self, model, target_layer, device=None):
        self.target_layer = target_layer

        if device is not None:
            model = model.to(device)
            self.model = model
        else:
            device = next(model.parameters()).device
            self.model = model
            self.device = device

        self.model.eval()

        # Hook for the feature maps and gradients
        self.features = SaveFeatures(self.target_layer)

    def forward(self, x):
        return self.model(x)
    
    def __call__(self, input_images, target_classes):
        # Ensure target_classes is a torch tensor
        if not isinstance(target_classes, torch.Tensor):
            target_classes = torch.tensor(target_classes).to(input_images.device)
        
        # Forward pass
        # Ensure the input tensor requires gradients
        input_images = input_images.to(self.device)
        input_images.requires_grad = True
        
        # Forward pass
        output = self.forward(input_images)
        
        # Zero all existing gradients
        self.model.zero_grad()
        
        # Create a one-hot vector for the target classes
        one_hot_output = torch.zeros_like(output)
        for i in range(target_classes.size(0)):
            one_hot_output[i, target_classes[i]] = 1
        
        # Backward pass to get the gradients
        output.backward(gradient=one_hot_output, retain_graph=True)

        # Get the gradients and feature maps
        gradients = self.features.gradients
        activations = self.features.features

        # Initialize the CAM for each image in the batch
        batch_size = input_images.size(0)
        cam_list = []
        
        for i in range(batch_size):
            # Grad-CAM++ specific calculations
            grad = gradients[i]
            act = activations[i]
            alpha = torch.clamp(grad, min=0.0)
            
            # Compute the weights (alpha_k^+)
            weights = torch.sum(alpha * act, dim=[1, 2]) / (torch.sum(act, dim=[1, 2]) + 1e-8)

            # Calculate Grad-CAM++ map
            cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(self.device)
            for j in range(weights.size(0)):
                cam += weights[j] * act[j, :, :]

            # Apply ReLU to the CAM
            cam = F.relu(cam)
            # Normalize the CAM to range [0, 1]
            cam = cam - cam.min()
            cam = cam / cam.max()

            # Upsample the CAM to match the input image size
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(input_images.size(2), input_images.size(3)), mode='bilinear', align_corners=False)
            cam = cam.squeeze(0).squeeze(0)

            # Convert CAM to numpy and store in list
            cam_list.append(cam.detach().cpu().numpy())
        
        # Return the CAMs for the batch and the target classes
        return np.array(cam_list)


# custom implementation of the gradcam

class GradCAM:
    def __init__(self, model, target_layer,device=None):
        self.target_layer = target_layer

        if device is not None:
            model=model.to(device)
            self.model=model
            
        
        else:
            device=next(model.parameters()).device
            self.model=model
            self.device=device

        self.model.eval()

        # Hook for the feature maps and gradients
        self.features = SaveFeatures(self.target_layer)

    def forward(self, x):
        return self.model(x)
    
    def __call__(self, input_images, target_classes):
        # Ensure target_classes is a torch tensor
        if not isinstance(target_classes, torch.Tensor):
            target_classes = torch.tensor(target_classes).to(input_images.device)
        
        # Forward pass
        output = self.forward(input_images.to(self.device))
        
        # Zero all existing gradients
        self.model.zero_grad()
        
        # Create a one-hot vector for the target classes
        one_hot_output = torch.zeros_like(output)
        for i in range(target_classes.size(0)):
            one_hot_output[i, target_classes[i]] = 1
        
        # Backward pass to get the gradients
        output.backward(gradient=one_hot_output, retain_graph=True)

        # Get the gradients and feature maps
        gradients = self.features.gradients
        activations = self.features.features

        # Initialize the CAM for each image in the batch
        batch_size = input_images.size(0)
        cam_list = []
        
        for i in range(batch_size):
            # Global Average Pooling on the gradients for the i-th image
            weights = torch.mean(gradients[i], dim=[1, 2])
            cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(self.device)

            # Weighted sum of the activations for the i-th image
            for j, w in enumerate(weights):
                cam += w * activations[i, j, :, :]

            # Apply ReLU to the CAM
            cam = F.relu(cam)
            # Normalize the CAM to range [0, 1]
            cam = cam - cam.min()
            cam = cam / cam.max()

            # upsample
            # Upsample the CAM to match the input image size
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(input_images.size(2), input_images.size(3)), mode='bilinear', align_corners=False)
            cam = cam.squeeze(0).squeeze(0)

            # Convert CAM to numpy and store in list
            cam_list.append(cam.detach().cpu().numpy())
        
        # Return the CAMs for the batch and the target classes
        return np.array(cam_list)


# Custom implementation of the SmoothGrad
class SmoothGrad():
    """
    custom implementation of the smooth grad based on the 
    WAM implementation

    values for the number of samples and the standard deviation 
    are based on the original paper
    """

    def __init__(self,
                 model,
                 n_samples=50,
                 stdev_spread=0.25,
                 random_seed=42,
                 device=None
                 ):


        self.model=model
        self.n_samples=n_samples
        self.stdev_spread=stdev_spread
        self.random_seed=random_seed

        if device is not None:
            model=model.to(device)
            self.model=model
        
        else:
            device=next(model.parameters()).device
            self.model=model
            self.device=device

    

    def compute_gradients(self,x,y):
        """
        computes the gradients with respect to the labels 
        for the batch of images passed as input.

        returns a [n, image_size, image_size] array
        where each coordinates indicates the gradient
        of the model wrt to this pixel
        """

        # inference, computation of the loss
        # and differentiation
        output=self.model(x.to(self.device))
        loss=torch.diag(output[:,y]).mean()
        loss.backward()

        # conversion as a numpy array
        return x.grad.detach().cpu().numpy().mean(axis=1)

    def __call__(self, x,y):
        """
        main implementation
        """
        np.random.seed(self.random_seed)

        # array that aggregates the averaged gradients
        avg_gradients=np.zeros((x.shape[0], x.shape[2], x.shape[3]))

        for _ in range(self.n_samples):
            
            noisy_x=torch.zeros(x.shape)

            for i in range(x.shape[0]):

                max_x=x[i,:,:,:].max()
                min_x=x[i,:,:,:].min()

                stdev=self.stdev_spread*(max_x-min_x).item()
                # generate noise calibrated for the current image
                noise = torch.normal(0, stdev, size=x[i].shape, dtype=torch.float32, device=x.device)

                # apply the noise to the images
                noisy_x[i,:,:,:]=x[i,:,:,:]+noise

            # compute the smoothgrad
            noisy_x.requires_grad_()
            avg_gradients+=self.compute_gradients(noisy_x,y)

        # compute the mean
        for k in range(avg_gradients.shape[0]):
            avg_gradients[k,:,:] /= self.n_samples

        return avg_gradients

    
# helper functions for the 2D case (computation of the mu fidelity, insertion and deletion)

def yield_image_baseline(img, masks):
    """
    Returns a generator of reconstructed images following the pattern 
    in the masks array.

    Reconstructs images by applying the averaged mask to each channel 
    and normalizing.

    Yields images one by one to reduce memory usage.
    """

    # Ensure img is in the correct data type to save memory
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    # Process masks one by one to save memory
    for i in range(masks.shape[0]):
        # Apply the mask to the image
        masked_img = masks[i, :, :, np.newaxis] * img
        
        # Normalize and convert to uint8
        masked_img_normalized = (normalize_data(masked_img) * 255).astype(np.uint8)
        
        # Convert to PIL Image and yield
        yield Image.fromarray(masked_img_normalized)



def reconstruct_images_baseline(img,masks):
    """
    wrapper of the former, 
    """
    return list(yield_image_baseline(img, masks))



def sum_importance(wam,indices,grid_size,n_samples, batch_size=None):
    """
    computes the importance of the wam at the specified suprepixel
    indices
    """

    if batch_size is None:
        batch_size=n_samples

    # convert the indices as masks over superpixels
    masks=np.zeros((n_samples,grid_size,grid_size), dtype=np.uint8)
    for i,index_set in enumerate(indices):
        x,y=zip(*index_set)
        masks[i,x,y]=1

    # upsample the masks
    zoom_factor=(1,wam.shape[0]/grid_size,wam.shape[1]/grid_size)

    importances=np.empty(n_samples)
    nb_batch=int(np.ceil(n_samples/batch_size))

    for batch_index in range(nb_batch):

        start_index=batch_index*batch_size
        end_index=min(n_samples,(batch_index+1)*batch_size)

        upsampled_mask=zoom(masks[start_index:end_index],zoom_factor,order=0)
        importances[start_index:end_index]=np.sum(wam*upsampled_mask, axis=(1,2))

        del upsampled_mask

    del masks
    return importances

def evaluate(x, y, model, batch_size):
    """
    Inference loop that returns the vector of predicted probabilities.

    x should be a copy of the same image.
    y should be a scalar, the ground truth label for image x.
    """

    device = next(model.parameters()).device
    model.eval()

    # Determine the number of batches
    nb_batch = int(np.ceil(len(x) / batch_size))
    out = np.empty(len(x), dtype=np.float32)

    with torch.no_grad():
        for batch_index in range(nb_batch):
            start_index = batch_index * batch_size
            end_index = min(len(x), (batch_index + 1) * batch_size)

            batch_x = x[start_index:end_index].to(device)

            # Forward pass
            preds = model(batch_x)
            preds_cpu = preds.cpu().detach().numpy()  # Move to CPU and convert to numpy

            # Compute probabilities
            probs = np.exp(preds_cpu) / np.sum(np.exp(preds_cpu), axis=1, keepdims=True)

            # Retrieve the probabilities for the y_th column (ground truth label)
            out[start_index:end_index] = probs[:, y]

            # Clean up variables
            torch.cuda.empty_cache()
            del batch_x, preds, preds_cpu, probs

    return out

def normalize_data(data):
    data=data.astype(np.float32)
    return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

def compute_auc(probs):
    """
    computes the AUC given the input vector

    normalizes the x and y axis (by construction)
    to divide by an area of 1

    takes as input the vector of predicted probs of size
    (1, n_features)

    returns the auc
    """
    
    y_scale = np.max(probs)
    x_scale = len(probs)
    
    return sum(probs) / (y_scale * x_scale)

def generate_masks(n_iter,wam):
    """
    generate the insertion masks and the deletion masks
    based on the wam
    
    sorts importance by decreasing order and generate masks 
    that contain increasingly more components (insertion)
    and increasingly less components (deletion)

    returns a tuple of arrays [n_iter,shape,shape] with the insertion and the deletion
    masks
    parameter:n_iter
    """
    # sort the values
    flat_indices = np.argsort(wam, axis=None)[::-1]  # Returns the indices that would sort the flattened array

    # Step 2: Convert flat indices to 2D coordinates
    coordinates = np.unravel_index(flat_indices, wam.shape)

    # Step 3: Pair the coordinates together (row, col) and zip them with sorted values
    sorted_coords = list(zip(coordinates[0], coordinates[1]))

    # geenrate the masks 
    n_components=int(len(sorted_coords)/n_iter)

    insertion_masks=np.zeros((n_iter+1,wam.shape[0],wam.shape[1]))
    deletion_masks=np.ones((n_iter+1,wam.shape[0],wam.shape[1]))

    for i in range(n_iter):
        corresponding_coords=sorted_coords[:min((i+1)*n_components, len(sorted_coords))]

        # retrieve the indices
        x=[c[0] for c in corresponding_coords]
        y=[c[1] for c in corresponding_coords]

        # for the insertion: set to 1
        insertion_masks[i+1,x,y]=1
        deletion_masks[i+1,x,y]=0


    # by construction, the last masks contains everything 
    insertion_masks[-1]=np.ones((wam.shape[0],wam.shape[1]))
    deletion_masks[-1]=np.zeros((wam.shape[0],wam.shape[1]))

    return insertion_masks, deletion_masks

def reconstruct_images(img,J,masks,wavelet="haar"):
    """
    returns a list of reconstructed images following the pattern
    in the masks array

    returns a list of images
    """

    reconstructed_images=[]

    for i in range(masks.shape[0]): # loop over the masks

        reconstructed_channels=[]

        # Process each color channel
        for j in range(3):
            # Extract the j-th channel
            channel_img = img[:, :, j]
            
            # Perform wavelet transform on the channel
            coeffs = pywt.wavedec2(channel_img, wavelet, level=J)
            arr, coeff_slices = pywt.coeffs_to_array(coeffs)
            
            # Apply the perturbation mask to the wavelet coefficients
            perturbed_wt = arr * masks[i,:,:]
            
            # Convert the perturbed array back to the coefficients list
            perturbed_coeffs = pywt.array_to_coeffs(perturbed_wt, coeff_slices, output_format='wavedec2')
            
            # Reconstruct the channel from the perturbed coefficients
            reconstructed_channel = pywt.waverec2(perturbed_coeffs, wavelet)
            reconstructed_channels.append(reconstructed_channel)

        reconstructed_images.append(
            np.stack(reconstructed_channels, axis=2)
        )

    return [Image.fromarray((normalize_data(rim)*255).astype(np.uint8)) for rim in reconstructed_images]


def generate_images_baseline(img,explanation,n_iter,mode):
    """
    generate a set of perturbation mask and returns a list of images reconstructed
    using the filtered attribution (in the pixel domain only)
    
    returns a list of images
    """
    # generate the masks
    insertion_masks,deletion_masks=generate_masks(n_iter,explanation)

    if mode=="insertion":
        reconstructed_images=reconstruct_images_baseline(img,insertion_masks)

    elif mode=="deletion":
        reconstructed_images=reconstruct_images_baseline(img,deletion_masks)

    return reconstructed_images

def generate_images(img,wam,J, n_iter, mode, wavelet="haar"):
    """
    generates a set of perturbation masks and returns a list of images 
    reconstructed using the filtered wam

    mode is a string that specifies whether it should be 
    reconstructed using insertion or deletion
    
    returns a list of images
    """
    # generate the masks
    insertion_masks,deletion_masks=generate_masks(n_iter,wam)

    if mode=="insertion":
        reconstructed_images=reconstruct_images(img,J,insertion_masks,wavelet=wavelet)

    elif mode=="deletion":
        reconstructed_images=reconstruct_images(img,J,deletion_masks,wavelet=wavelet)

    return reconstructed_images

def generate_subsets(grid_size, subset_size, sample_size):
    """
    Generates a list of two-dimensional indices 
    of size subset_size (<= grid_size x grid_size)
    that correspond to the set of features to investigate.
    The list is generated as a sample of the subsets of size subset_size.
    """
    coordinates = [
        [
            (i // grid_size, i % grid_size)
            for i in random.sample(range(grid_size * grid_size), subset_size)
        ]
        for _ in range(sample_size)
    ]
    return coordinates
