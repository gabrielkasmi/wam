# -*- coding: utf-8 -*-

# libraries

# Class that defines the wam for 3D shapes. The "base" computes the gradient. As it returns a noisy "saliency map", we 
# apply the SmoothGrad technique to obtain better explanations [Not implemented yet]

from scipy import interpolate
import numpy as np
import ptwt
import torch
from scipy.ndimage import zoom

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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


    if dimension==3: # case of shapes
        if grads:
            numpy_coeffs=[coeffs[0].grad.detach().squeeze().cpu().numpy()]

            for detail in coeffs[1:]: # iterate over the levels
                numpy_coeffs.append(
                    {k:v.grad.detach().squeeze().cpu().numpy() for k,v in detail.items()}
                )
        else:
            numpy_coeffs=[coeffs[0].detach().squeeze().cpu().numpy()]

            for detail in coeffs[1:]: # iterate over the levels
                numpy_coeffs.append(
                    {k:v.detach().squeeze().cpu().numpy() for k,v in detail.items()}
                )
                
    return numpy_coeffs



def filter_coeffs(coeffs, EPS, normalized=False):
    """filters the input coeff"""

    if normalized:
        return (coeffs >= EPS).astype(int)
    else:
        normalized_coeffs= (coeffs-coeffs.min()) / ((coeffs.max() - coeffs.min()))
        
        return (normalized_coeffs > EPS).astype(int)


class BaseWAM3D():
    """
    Implements the base wam for 3D signals

    The instance specifies whether the input data are voxels
    or point clouds. In the case of point clouds, a 1D wavelet transform
    is applied to each dimension.
    """
    def __init__(self,
                 model, # model
                 wavelet="haar", # wavelet instance
                 J=1, # number of decomposition levels
                 approx_coeffs=False,
                 device=None,
                 mode="symmetric",
                 instance="voxels",
                 normalize=True,
                 EPS=0.451
                 ):
        
        self.wavelet=wavelet
        self.J=J
        self.approx_coeffs=approx_coeffs
        self.mode=mode
        self.instance=instance
        self.EPS=EPS
        self.normalize=normalize
        self.input_size = None

        if device is not None:
            model=model.to(device)
            self.model=model
            self.device=device
        
        else:
            device=next(model.parameters()).device
            self.model=model
            self.device=device

    def refactor(self, coeffs, input_size=16):

        """
        reshapes the detached coefficients as a cube
        summarizing all the levels and all the orientations
        """
        if self.input_size is None:
            self.input_size=input_size
        # initialize the array that will be returned
        dyadic_transform=np.empty((len(coeffs),self.input_size,self.input_size,self.input_size), dtype=np.float32)

        # indices of the levels 
        level_indices=[int(self.input_size / 2**j) for j in range(self.J+1)][::-1]
        level_indices.insert(0,0)

        for k, input_coeff in enumerate(coeffs):

            for i in range(self.J+1):
                start_index=level_indices[i]
                end_index=level_indices[i+1]

                # approximation coefficients
                if start_index==0: # approximation coefficients
                    dyadic_transform[k,:end_index, :end_index, :end_index]=np.abs(input_coeff[i])

                else:
                    # for each level, store the values of the different orientations
                    # aad, ada, add, dad, dda, add, ddd
                    dyadic_transform[k,start_index:end_index,start_index:end_index, start_index:end_index]=np.abs(input_coeff[i]["ddd"])

                    dyadic_transform[k,:start_index,:start_index, start_index:end_index]=np.abs(input_coeff[i]["aad"])
                    dyadic_transform[k,:start_index,start_index:end_index, :start_index]=np.abs(input_coeff[i]["ada"])
                    dyadic_transform[k,:start_index,start_index:end_index, start_index:end_index]=np.abs(input_coeff[i]["add"])

                    dyadic_transform[k,start_index:end_index,:start_index, :start_index]=np.abs(input_coeff[i]["daa"])
                    dyadic_transform[k,start_index:end_index, :start_index, start_index:end_index]=np.abs(input_coeff[i]["dad"])
                    dyadic_transform[k,start_index:end_index, start_index:end_index, :start_index]=np.abs(input_coeff[i]["dda"])


        return dyadic_transform

    def evaluate_voxels(self,x,y,shape=True):
        """
        computation of the wam for voxels (i.e. 3D shapes)

        natively returns a 'cube' (which generalizes the wavelet transform)
        from lower dimensions of shape
        
        return a cube with the gradients and a cube with the coeffs

        (batch_size, input_size, input_size, input_size) where each cube
        corresponds to coefficients at a given level and orientation
        """

        # compute the wavelet transform 
        # of the input
        #if permute is not None: # permute according to the tuple passed as input
        #    x=x.permute(permute)

        # store the size of the inputs for the
        # reconstruction of the wavelet transform
        if shape:
            
            self.input_size=x.shape[-1]
            grads=[]
            reconstructions=[]
            for index in range(x.shape[0]):
                coeffs=ptwt.wavedec3(x[index], self.wavelet, level=self.J, mode=self.mode)
                
                # require the gradients
                grad_coeffs=[coeffs[0].requires_grad_()]
                for coeff in coeffs[1:]:
                    grad_coeff={
                        k:v.requires_grad_() for k,v in coeff.items()
                    }
                    grad_coeffs.append(grad_coeff)

                # invert
                grads.append(grad_coeffs)
                reconstructions.append(ptwt.waverec3(grad_coeffs,self.wavelet))

            x_grad=torch.stack([t for t in reconstructions]).to(self.device)

        else:
            # require the gradients directly from the coefficients
            grads=[]
            reconstructions=[]

            for x_item in x:
                grad_coeffs=[x_item[0].requires_grad_()]
                for coeff in x_item[1:]:
                    grad_coeff={k:v.requires_grad_() for k,v in coeff.items()}
                    grad_coeffs.append(grad_coeff)

                grads.append(grad_coeffs)
                reconstructions.append(ptwt.waverec3(grad_coeffs,self.wavelet))
            x_grad=torch.stack([t for t in reconstructions]).to(self.device)

        # evaluate
        if y is None: # case where we work with the Nerf grids and the ResNet's representation
            preds=self.model(x_grad.unsqueeze(0))# .squeeze() # remove all dimensions equal to 1
                                                            # if there is a single scene, should
                                                            # return a [2048] tensor

            # compute the gradients
            preds.mean().backward()
        
        else:
            outputs=self.model(x_grad)

            loss=torch.diag(outputs[:, y]).mean()
            loss.backward()
            
        # detach and return the gradients
        detached_grads = [to_numpy(grad, 3, grads=True) for grad in grads]
        detached_coeffs = [to_numpy(grad, 3, grads=False) for grad in grads]
        self.coeffs=detached_coeffs

        return self.refactor(detached_grads)

    def evaluate_point_clouds(self, x,y, permute):
        """
        applies the 1D wavelet transform to each dimension independently
        and evaluate the sensitivity of the model wrt these coefficients.

        takes as input a point cloud of shape [batch_size, n_points, n_dims]
        and returns the coefficients
        """

        # compute the wavelet transform of the point cloud
        # and require the gradients
        # Wavelet decomposition and reconstruction
        result_list = []
        coeffs_out=[]

        self.batch_size=x.shape[0]
        self.shape_size=x.shape[1]
        self.input=x.detach().cpu().numpy()

        # # map en 1D
        # prime_numbers =[73856093, 19349669, 83492791]
        # x[...,0] = x[...,0] * prime_numbers[0]
        # x[...,1] = x[...,1] * prime_numbers[1]
        # x[...,2] = x[...,2] * prime_numbers[2]
        # # # print min max for each channel
        # # # create a zero array of size [batch_size, n_points, 1]
        # x_mono = torch.zeros(x.shape[0], x.shape[1], 1)
        # # x_mono[:, :, 0] = x[:, :, 0] ^ x[:, :, 1] ^ x[:, :, 2]
        # x_mono = x[:, :, 0] + x[:, :, 1] + x[:, :, 2]
        # x_mono = x_mono.unsqueeze(-1)

        x_flatten = x.view(self.batch_size, -1)
        x_mono = x_flatten.unsqueeze(-1)

        # ondelettes + grad

        coeffs = ptwt.wavedec(x_mono[:, :, 0], self.wavelet, level=self.J)
        grad_coeffs = [c.requires_grad_() for c in coeffs]

        # inverse
        reconstructed = ptwt.waverec(grad_coeffs, self.wavelet) # B, N, 1

        # map en 3D
        # inverse x_mono to x (B, N, 1 => B, N, 3)
        x_grad = reconstructed.view(x.shape[0], x.shape[1], 3)


        # test ---
        # go from ch = 3 to ch = 1
        # prime_numbers =[73856093, 19349669, 83492791]
        # x[...,0] = x[...,0] * prime_numbers[0]
        # x[...,1] = x[...,1] * prime_numbers[1]
        # x[...,2] = x[...,2] * prime_numbers[2]
        # # print min max for each channel
        # # create a zero array of size [batch_size, n_points, 1]
        # x_mono = torch.zeros(x.shape[0], x.shape[1], 1)
        # # x_mono[:, :, 0] = x[:, :, 0] ^ x[:, :, 1] ^ x[:, :, 2]
        # x_mono = x[:, :, 0] + x[:, :, 1] + x[:, :, 2]
        # x = x_mono.unsqueeze(-1)
        # x = x.repeat(1, 1, 3)
        # ---

        # for dim in range(x.shape[2]):
        #     # Perform wavelet decomposition
        #     coeffs = ptwt.wavedec(x[:, :, dim], self.wavelet, level=self.J)
            
        #     # Convert coefficients to tensors
        #     grad_coeffs = [c.requires_grad_() for c in coeffs]
        #     coeffs_out.append(coeffs)
            
        #     # Reconstruct tensor (ensure this does not break the gradient flow)
        #     reconstructed = ptwt.waverec(grad_coeffs, self.wavelet)
            
        #     # Append to list
        #     result_list.append(reconstructed.unsqueeze(-1))  # Add extra dimension for concatenation

        # x_grad = torch.cat(result_list, dim=-1)
        # inference with the reconstructed point cloud
        # assumed to work with a point net that returns
        # a tuple 
        if permute is not None:
            output, _,_ = self.model(x_grad.permute(permute).to(self.device))
            #output, _,_ = self.model(x.requires_grad_().permute(permute).to(self.device))


        else:
            output = self.model(x_grad.to(self.device))

        # computation of the loss and of the gradients

        loss = torch.diag(output[:,y]).mean()
        loss.backward()

        # retrieve the coefficients and the gradients

        detached_coeffs=to_numpy(coeffs, 1, grads=False)
        detached_grads=to_numpy(coeffs, 1, grads=True)

        #detached_grads=[],[]

        # for dim in range(x.shape[2]):

        #     detached_coeffs.append(
        #         to_numpy(coeffs_out[dim], 1, grads=False)
        #     )
        #     detached_grads.append(
        #         to_numpy(coeffs_out[dim], 1, grads=True)
        #     )

        # # TODO. Represent the gradients in a convenient way
        
        return detached_coeffs,detached_coeffs#self.filter_point_clouds(detached_coeffs,detached_grads)

    def __call__(self,x,y=None,permute=None, shape=True): # performed only on x, as we 
                     # compute the gradients based on the 
                     # representation of the ResNet3D
        """
        computes the gradients on the wavelet coefficients of the
        shape. 

        add as new attributes the gradients and the coefficients
        and returns the gradients as a np.array.

        In the case of point clouds, the gradients are returned as a list
        in the wavelet domain and as a filtered point cloud, where only the 
        coefficients that exceed a given value are returned.

        the threshold value can be set by the user when initializing the 
        explainer. 
        """

        if self.instance=="voxels":
            return self.evaluate_voxels(x,y,shape=shape)
        
        elif self.instance=="point_clouds":
            print('Not implemented yet')
            #return self.evaluate_point_clouds(x,y, permute)
    
    def filter_point_clouds(self, coeffs, grads):

        """
        TODO. Edit this function to match the evaluate_voxel format. 
        reconstruct the point cloud using only the wavelet coefficients
        whose gradient exceeds a given threhsold EPS. 

        by default, a level-wise normalization of the coefficients between 0 and 1 is applied.

        returns a point cloud of shape [batch_size, n_points, n_dim]
        """
        # create a time-frequency plot for each dimension
        time_frequency_mask=np.empty((self.batch_size, 3,self.J+1,self.shape_size))
        target_indices=np.linspace(0,1,self.shape_size)

        reconstructed_shape=[]
        for dim in range(len(grads)):

            dim_grad, dim_coeffs=grads[dim], coeffs[dim]

            # loop across the levels
            level_filter=[]
            for level in range(len(dim_grad)):

                level_grad=dim_grad[level]
                level_coeff=dim_coeffs[level]

                # at each level, upsample the time series to match
                # the full length of the series.
                source_indices=np.linspace(0,1,level_grad.shape[1])

                upsampled_grad=np.array([
                    interpolate.interp1d(source_indices, g, kind="cubic")(target_indices) for g in level_grad
                ])

                time_frequency_mask[:,dim,level,:]=upsampled_grad

            # once the mask has been generated, sum it across the frequency
            # dimension
            time_frequency_values=np.sum(time_frequency_mask, axis=(1,2))
            time_frequency_values=normalize(time_frequency_values)

            # retain the values that exceed the threshold
            indices=[np.where(abs(ts) > self.EPS)[0] for ts in time_frequency_values]

            # reconstruct the shapes based on the indices
            reconstructed_shape=[]
            for i, index in enumerate(indices):
                reconstructed_shape.append(self.input[i,index,:].transpose(0,1))
                
        return reconstructed_shape, time_frequency_values

        #return filtered_point_clouds
        
    def filter_voxels(self, normalized=True):
        """
        reconstruct a filtered shape based on the filtering of the 
        normalized gradient coefficients

        returns an array of shape (n_samples, n_channels, x,y,z) of filtered shapes
        
        filtering here consists in normalizing the wavelet coefficient
        
        """

        filtered_shapes=[]
        
        for grad, coeff in zip(self.grads, self.coeffs):

            # for one sample, grad and coeff are a list
            # where the 0th index is the approximation coefficients
            # and the 1: indices are the levels. 
            # in each level, we have a dictionnary with keys {aad, ada, daa, add, dad, dda, ...} 
            # which correspond to the detail coefficients at different orientations (similar to the vertical and diagonal
            # details in a 2D wavelet transform, where the coefficients are sometimes labeled as AD, DD, DA)

            # filter the coefficients
            approx_grad=(grad[0] - np.min(grad[0])) / (np.max(grad[0] - np.min(grad[0])))

            filtered_approx= torch.tensor(coeff[0] * approx_grad)
            filtered_details=[filtered_approx]
            
            for detail_g, detail_c in zip(grad[1:],coeff[1:]):
                loc_coeffs={}
                
                for orientation in detail_g.keys():
                    dg=detail_g[orientation]

                    # normalize the gradients in [0,1] (amounts to a soft thresholding)
                    #dg=(dg - np.min(dg)) / (np.max(dg) - np.min(dg))
                    dg=np.abs(dg)/dg.max()


                    dc=detail_c[orientation]
                    loc_coeffs[orientation]= torch.tensor(dc * (dg >= self.EPS))
                    print(dc.shape, dg.shape)


                    print(np.sum(dc * (dg >= self.EPS)))
                    
                filtered_details.append(loc_coeffs)
                    
                # once the details have been filtered, one needs to reconstruct the shape
                # reconstruction can be done if the filtered coefficients are formatted
                # properly (i.e. with the list(array, dict , ...)).

            filtered_shapes.append(ptwt.waverec3(filtered_details, wavelet=self.wavelet).numpy())
        
        filtered_shapes=np.array(filtered_shapes)

        return filtered_shapes #(filtered_shapes - filtered_shapes.min()) / (filtered_shapes.max() - filtered_shapes.min())     

# wrapper that runs the Basewam n_samples time to smooth the gradients
# smoothing is either through path integration or averaging across noisy samples

# averages are either path integration or 'true' averaging
class WaveletAttribution3D(BaseWAM3D):
    """
    wrapper that computes the smooth grad for 3D shapes
    """
    
    def __init__(self, 
                 model, 
                 wavelet="haar", # wavelet instance
                 J=3, # number of decomposition levels
                 approx_coeffs=False,
                 device=None,
                 mode="symmetric",
                 instance="voxels",
                 method="smooth",
                 normalize=True,
                 EPS=0.451,
                 n_samples=25, 
                 stdev_spread=0.0001, # range [.2-.3] produces the best results
                                # visually 
                 random_seed=42):
            super().__init__(model, 
                            wavelet=wavelet, # wavelet instance
                            J=J, # number of decomposition levels
                            approx_coeffs=approx_coeffs,
                            device=device,
                            mode=mode,
                            instance=instance,
                            normalize=normalize,
                            EPS=EPS)
            

            """
            model, # model
            wave="haar", # wavelet instance
            J=1, # number of decomposition levels
            approx_coeffs=False,
            device=None,
            mode="symmetric"
            """        
            self.n_samples=n_samples
            self.stdev_spread=stdev_spread
            self.random_seed=random_seed
            self.method=method

            self.wam=BaseWAM3D(model,wavelet=wavelet,J=J,mode=mode,
                                     device=device,approx_coeffs=approx_coeffs,
                                     instance=instance,normalize=normalize,EPS=EPS)


    def smooth(self,x,y=None, permute=None):
         """
         add noise and averages the predictions
         """

         np.random.seed(self.random_seed)

         # array that will store the outputs
         input_size=x.shape[-1] # we assume that the shape is a cube

         if self.input_size is None:
             self.input_size=input_size
         avg_gradients=np.zeros((x.shape[0],input_size, input_size, input_size),
                                dtype=np.float32)
                  
         for _ in range(self.n_samples):
              
            noisy_x=torch.zeros(x.shape)

            for i in range(x.shape[0]): # iterate over the batch of shapes
                

                max_x=x[i,0,:,:,:].max() # 1st dimension is 1
                min_x=x[i,0,:,:,:].min()

                stdev=self.stdev_spread*(max_x-min_x)
                # generate noise calibrated for the current image
                noise=np.random.normal(0,stdev,x.shape[2:]).astype(np.float32)
                # apply the noise to the images
                noisy_x[i,0,:,:,:]=x[i,0,:,:,:]+noise

            # compute the wam
            avg_gradients+=self.wam(noisy_x,y)
            
            
            for k in range(avg_gradients.shape[0]): # compute the mean
               
               avg_gradients[k,:,:]/=self.n_samples

         self.grads=avg_gradients

         return avg_gradients
               
               
    def alter(self, alpha, coeffs):
        """
        alters and returns the altered wavelet transform
        """

        altered_coeffs=[]

        for coeff in coeffs:
            tmp_coeffs=[coeff[0] * alpha]

            for l_coeffs in coeff[1:]:
                tmp_coeffs.append(
                    {k:v*alpha for k, v in l_coeffs.items()}
                )

            altered_coeffs.append(tmp_coeffs)
        
        return altered_coeffs

    
    def intergrated_wam(self,x,y=None,permute=None):
         """
         integrates the gradients to smooth the explaination
         """

         # retrieve the coeffs, corresponds to z
         coeffs=[ptwt.wavedec3(x[index], self.wavelet, level=self.J, mode=self.mode) for index in range(x.shape[0])]
         np_coeffs=[to_numpy(coeff,3,grads=False) for coeff in coeffs]
         baseline_z=self.refactor(np_coeffs, input_size=x.shape[-1])

         # generate alpha
         alphas=np.linspace(0,1,self.n_samples)

         # generate the path that will accumulate the gradients
         grad_path=np.empty((baseline_z.shape[0], self.n_samples, baseline_z.shape[1],baseline_z.shape[2], baseline_z.shape[3]),
                            dtype=np.float32)
         
         for i, alpha in enumerate(alphas):
             
             path_coeffs=self.alter(alpha, coeffs)
             grad_path[:,i,:,:,:]=self.wam(path_coeffs,y,shape=False)

         # once computed the path, average the integral using the 
         # trapezoidal rule
         integral=np.trapz(np.nan_to_num(grad_path), axis=1)

          # return the results

         self.grads=baseline_z*integral
         return baseline_z*integral




    def __call__(self,x,y=None,permute=None):
        """
        the current implementation will only work with 
        point cloud instances.

        returns the gradient cube averaged over the batch of samples
        """
        # set up the random seed 
        if self.method=="smooth":
             return self.smooth(x,y)
        
        elif self.method=="integratedgrad":
             return self.intergrated_wam(x,y)

    def visualize(self):
         """
         converts the smoothed gradient cube for visualization over 
         the point cloud

         all scales are upsamples to the size of the input shape

         returns a [n_batch, n_levels+1, shape, shape, shape] array
         where in the second dimension:
                the index 0 correspond to the approximation coefficients
                and the index 1 to n_levels +1 to the detail coefficients


         """

         # retrieve the gradients
         input_size=self.input_size # get the shape of the input
         gradients=self.grads

         level_indices=[int(self.input_size / 2**j) for j in range(self.J+1)][::-1]
         level_indices.insert(0,0)

         # set up the array that will contain the upsampled gradients
         visualizations=np.empty((gradients.shape[0], self.J+2,gradients.shape[1],gradients.shape[2],gradients.shape[3]),dtype=np.float32)

         for i in range(gradients.shape[0]):
             for j in range(self.J+1):
                 start_index=level_indices[j]
                 end_index=level_indices[j+1]


                 if start_index==0:
                     # retrieve the approximation coefficients
                     coeffs=gradients[i,:end_index,:end_index,:end_index]
                     zoom_factor=int(input_size/coeffs.shape[-1])
                     upsampled=zoom(coeffs,zoom_factor, order=1)
                     visualizations[i,j,:,:,:]=upsampled / upsampled.max()
                 else:
                     
                     # retrieve each level
                    aad=gradients[i,:start_index,:start_index, start_index:end_index]
                    ada=gradients[i,:start_index,start_index:end_index, :start_index]
                    add=gradients[i,:start_index,start_index:end_index, start_index:end_index]
                    daa=gradients[i,start_index:end_index,:start_index, :start_index]
                    dad=gradients[i,start_index:end_index, :start_index, start_index:end_index]
                    dda=gradients[i,start_index:end_index, start_index:end_index, :start_index]

                    common=add+ada+add+daa+dad+dda
                    zoom_factor=int(input_size/common.shape[-1])
                    upsampled=zoom(common,zoom_factor, order=1)
                    visualizations[i,j,:,:,:]=upsampled / upsampled.max()
         
         # last "level" corresponds to the sum of all levels

         all_levels=np.sum(visualizations[:,:self.J+1,:,:,:], axis=1)
         visualizations[:,-1,:,:,:]=all_levels/all_levels.max()
         
         return visualizations
