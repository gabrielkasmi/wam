# -*- coding: utf-8 -*-

import pywt
import numpy as np

def compute_levelized_masks(grad_wam,J):
    """
    returns a [J+1,W,H] array where each J 
    corresponds to the coefficients at the Jth level
    from the finest to the coarsest. 
    J+1 corresponds to the approximation coefficients
    """

    img_size=grad_wam.shape[0]

    partial_masks=np.zeros((J+1,img_size,img_size))

    for j in range(J):

        start_index=int(img_size/2**(j+1))
        end_index=int(img_size/2**j)

        # add the vertical, horizontal and diagonal
        # coefficients at each level
        partial_masks[j,start_index:end_index,start_index:end_index] = grad_wam[start_index:end_index,start_index:end_index]
        partial_masks[j,start_index:end_index,:start_index] += grad_wam[start_index:end_index,:start_index]
        partial_masks[j,:start_index,start_index:end_index] += grad_wam[:start_index,start_index:end_index]

    # add the approximation coefficients
    approx_index=int(img_size/2**J)
    partial_masks[J,:approx_index,:approx_index]=grad_wam[:approx_index,:approx_index]

    return partial_masks

def generate_partial_image(img,grad_wam,q,J, wavelet="haar"):
    """
    given an image and its gradwam, computes
    an image generated using the qth quantile of
    the most important coefficients

    we apply the same perturbation across the 
    channels of the input image
    the mask is binarized

    inputs
    img: a np.array of shape (C,W,H)
    grad_wam: a np.array of shape (W,H)

    returns a tuple with 
        a np.ndarray corresponding to the
        partial image 
        a np.ndarray corresponding to the filtered
        grad wam 
    """
    quantile_value = np.quantile(grad_wam, q)
    perturbation_mask=(grad_wam >= quantile_value)

    # Initialize a list to store the reconstructed channels
    reconstructed_channels = []

    # Process each color channel
    for j in range(3):
        # Extract the j-th channel
        channel_img = img[:, :, j]
        
        # Perform wavelet transform on the channel
        coeffs = pywt.wavedec2(channel_img, wavelet, level=J)
        arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        
        # Apply the perturbation mask to the wavelet coefficients
        perturbed_wt = arr * perturbation_mask
        
        # Convert the perturbed array back to the coefficients list
        perturbed_coeffs = pywt.array_to_coeffs(perturbed_wt, coeff_slices, output_format='wavedec2')
        
        # Reconstruct the channel from the perturbed coefficients
        reconstructed_channel = pywt.waverec2(perturbed_coeffs, wavelet)
        reconstructed_channels.append(reconstructed_channel)

    # Stack the reconstructed channels to form the final reconstructed image
    return np.stack(reconstructed_channels, axis=2), perturbation_mask * grad_wam

def generate_disentangled_images(grad_wam,image,J,EPS=0.1, wavelet="haar"):
    """
    computes the partial images using the important coefficients
    filtered at each level (+ the approximation coefficients). Sorts the 
    levels from the finest to the coarsest. 

    EPS is an offset parameter. The higher, the fewer coefficients.

    returns 
    partial_images: a np.ndarray of shape [J+1,W,H,C]
    filtered_masks : a np.ndarray of shape [J+1,W,H] 
    """

    # retrieve the filtered masks
    filtered_masks=compute_levelized_masks(grad_wam,J)

    print(filtered_masks.shape)
    img_size=image.shape[0]

    # initialize the array for the partial images
    partial_images=np.zeros((J+1,img_size,img_size,3))

    # invert the image based on the partial masks
    for l in range(filtered_masks.shape[0]):

        # Initialize a list to store the reconstructed channels
        reconstructed_channels = []

        # Process each color channel
        for j in range(3):
            # Extract the j-th channel
            channel_img = image[:, :, j]
            
            # Perform wavelet transform on the channel
            coeffs = pywt.wavedec2(channel_img, wavelet, level=J)
            arr, coeff_slices = pywt.coeffs_to_array(coeffs)
            
            # Apply the perturbation mask to the wavelet coefficients
            perturbed_wt = arr * (filtered_masks[l] > (filtered_masks[l].min() + EPS))

            # Convert the perturbed array back to the coefficients list
            perturbed_coeffs = pywt.array_to_coeffs(perturbed_wt, coeff_slices, output_format='wavedec2')
            
            # Reconstruct the channel from the perturbed coefficients
            reconstructed_channel = pywt.waverec2(perturbed_coeffs, wavelet)
            reconstructed_channels.append(np.clip(reconstructed_channel, 0, 255))
            
        reconstructed_array=np.stack(reconstructed_channels, axis=2)
        # Stack the reconstructed channels to form the final reconstructed image
        partial_images[l,:,:,:]=reconstructed_array

    return partial_images, filtered_masks