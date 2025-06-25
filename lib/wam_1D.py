# -*- coding: utf-8 -*-

# libraries

import torch
import numpy as np
import ptwt
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import pywt
import librosa
import tqdm
import pywt

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

class BaseWAM1D():
    def __init__(self,
                 model,
                 wavelet="haar",
                 J=2,
                 mode="symmetric",
                 device=None,
                 approx_coeffs=False,
                 n_mels=128,
                 n_fft=1024,
                 sample_rate=44100,
                 ):
        
        self.wavelet=wavelet
        self.J=J
        self.mode=mode
        self.approx_coeffs=approx_coeffs


        # parameters for the melspec
        self.n_mels=n_mels
        self.n_fft=n_fft
        self.sample_rate=sample_rate

        if device is not None:
            model=model.to(device)
            self.model=model
            self.device=device
        
        else:
            device=next(model.parameters()).device
            self.model=model
            self.device=device

    def __call__(self,x,y,rates=None, waveform=True):
        """
        computes the explanations for the input x 
        with respect to its labels y.

        x: should be a list(array) or a tensor of shape 
        [N,W] where N is the batch size and W the size of the
        waveform

        rates (opt, list): should be a list that records the bitrate
        of the waveform. If none, then the default 44100 is applied.
        """
        # add the list of rates
        self.rates=rates

        if waveform:
            # compute the wavelet transform of the sound wave
            if isinstance(x, list):
                x=torch.tensor(np.array([wf/wf.max() for wf in x]).astype(np.float32))

            # compute the coefficients
            coeffs=ptwt.wavedec(x, self.wavelet,level=self.J,mode=self.mode)

        else: 
            coeffs=x
        # require the gradients
        grad_coeffs=[c.requires_grad_() for c in coeffs]

        # reconstruct with the gradients
        x_grad=ptwt.waverec(grad_coeffs, self.wavelet)

        # compute the melspec
        melspecs=self.compute_melspec(x_grad, n_fft=self.n_fft, sample_rate=self.sample_rate, n_mels=self.n_mels)
        melspecs.retain_grad()
        # inference and gradients
        output = self.model(melspecs.to(self.device))
 
        loss = torch.diag(output[:,y]).mean()
        loss.backward()

        # append here the melspec computed with the gradients
        # and retain its gradients
        # remove the unecessary dimension to return a [N,W,H] array
        # self.melspec=melspecs.detach().cpu().numpy().squeeze()

        # append the coefficients
        self.wavelet_coeffs=to_numpy(coeffs,dimension=1,grads=False)

        # append the 
        self.gradient_coeffs=to_numpy(grad_coeffs,dimension=1,grads=True)

        # equivalent of the gradcam, 
        #self.melspec_gradients=melspecs.grad.detach().cpu().numpy().squeeze()


        # additional stuff here that can be interesting: reconstruct 
        # the sound using the important components (across scales)
        
        # do stuff here ...

        # summarizes the important areas with the pseudo scaleogram
        # derived from the gradients
        return melspecs.grad.detach().cpu().numpy().squeeze(), to_numpy(grad_coeffs,dimension=1,grads=True)

    def visualize_grad_wam(self, coeffs):
        """
        plots the gradient on the "scaleogram" of the signal
        we consider the "pseudo" scaleogram as we do not repeat the lowest scales
        this representation is a helper for more convenient picking of important components
        in the signal
        """
                
        batch_size=coeffs[0].shape[0]
        max_length=coeffs[-1].shape[1] # defines the size of the window

        coeffs_per_samples=[]

        for i in range(batch_size):
            coeffs_per_samples.append(
                [coeff[i] for coeff in coeffs]
            )

        # define the scalogram matrix
        # fill it with nans 
        scalograms=np.ones((batch_size,self.J+1,max_length)) * np.nan

        for i, samples in enumerate(coeffs_per_samples):

            # plot the approximation coefficients
            approx=np.abs(samples[0]) 
            approx/=approx.max()
            details=samples[1:]

            boundary=approx.shape[0]
            scalograms[i,0,:boundary]=approx

            for j, detail in enumerate(details):

                detail=np.abs(detail)
                detail /= detail.max()

                boundary=detail.shape[0]
                scalograms[i,j+1,:boundary]=detail
        
        return scalograms

    def compute_melspec(self, reconstruction, n_fft=1024,sample_rate=44100,n_mels=128):
        """
        compute the melspec of an input batch of tensors
        reconstructed from the wavelet transform with grads

        inputs:
        power: if we keep the power scale or decibels
        reconstruction: the reconstruced soundwaves of shape [N,W]
        sample_rate, n_fft, n_mels: the parameters for the MelSpect
        
        returns a [N,W,H] tensor
        """
        # instantiate the Melspect and the power_to_db conversion
        power_to_db=AmplitudeToDB()
        mel_spectrogram=MelSpectrogram(sample_rate=sample_rate,n_fft=n_fft,n_mels=n_mels)
        
        melspecs=[]

        for waveform in reconstruction:
                
                melspec=mel_spectrogram(waveform)
                melspecs.append(
                    power_to_db(melspec).T.squeeze(-1).unsqueeze(0)
                )
        
        return torch.stack(melspecs)
    
    def filter(self, EPS):
        """
        function that filters the signal keeping only the values
        for which the gradient is above EPS. 

        returns a list of reconstructed waveforms. Its length is the batch size.
        """

        # retrieve the gradients
        # and the wavelet coefficients more conveniently
        gradients=self.gradient_coeffs
        coefficients=self.wavelet_coeffs

        # define the mask
        # filters the normalized absolute value of the gradient
        masks=[
            (np.abs(grads) / grads.max() > EPS).astype(int) for grads in gradients
        ]

        filtered_coeffs=[
            coeff * mask for coeff, mask in zip(coefficients,masks)
        ]

        filtered_sounds=pywt.waverec(filtered_coeffs, wavelet=self.wavelet)

        return filtered_sounds 


class WaveletAttribution1D(BaseWAM1D):
    def __init__(self,
                 model,
                 wavelet="haar",
                 J=3,
                 method="smooth",
                 mode="reflect",
                 device=None,
                 approx_coeffs=False,
                 n_mels=128,
                 n_fft=1024,
                 sample_rate=44100,
                 n_samples=25, 
                 stdev_spread=0.001, # after visual inspection
                 random_seed=42
                 ):
            super().__init__(model, 
                             wavelet=wavelet, 
                             J=J, 
                             device=device,
                             mode=mode,
                             approx_coeffs=approx_coeffs,
                             n_mels=n_mels,
                             n_fft=n_fft,
                             sample_rate=sample_rate)
            """
            self.model,
            self.wavelet="haar",
            self.J=2,
            self.mode="reflect",
            self.device=None,
            self.approx_coeffs=False,
            self.n_mels=128,
            self.n_fft=1024,
            self.sample_rate=44100,
            """

            self.n_samples=n_samples
            self.stdev_spread=stdev_spread
            self.random_seed=random_seed
            self.method=method
            self.wam=BaseWAM1D(
                model, wavelet=wavelet, J=J, mode=mode, device=device, approx_coeffs=approx_coeffs, n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate
            )

    def smooth_wam(self, x,y):
        """
        smoothgrad based implementation of the wam
        """

        if isinstance(x,list): # convert the input as tensor if necessary
            x=torch.tensor(
                np.array([wf/wf.max() for wf in x]).astype(np.float32)
            )

        # initialize the noise
        np.random.seed(self.random_seed)

        melspecs=[]
        gradients=[]

        # iterate over the waveforms of the batch
        for _ in range(self.n_samples):
            
            # generate noise
            noisy_x=torch.zeros(x.shape)

            for i in range(x.shape[0]): # iterate over the batch
                max_x=x[i].max() # data is unidimensional 
                min_x=x[i].min()

                stdev=self.stdev_spread*(max_x-min_x)
                noise=np.random.normal(0,stdev,x.shape[1]).astype(np.float32)
                noisy_x[i]=x[i]+noise
            
            # compute the explanations
            # retrieve the melspecs
            melspec, grads=self.wam(noisy_x,y)

            melspecs.append(melspec)
            gradients.append(grads)


        # average the gradients level-wse
        avg_gradients=[]
        for j in range(self.J+1):
            avg=np.mean(
                np.array([grad[j] for grad in gradients]), axis=0
            )
            avg_gradients.append(avg)

        self.melspecs=np.mean(np.array(melspecs),axis=0)
        self.grad_coeffs=avg_gradients

        return np.mean(np.array(melspecs),axis=0), avg_gradients

    def alter(self,alpha,coeffs):

        altered_coeffs=[]
        for coeff in coeffs:
            altered_coeffs.append(alpha * coeff)

        return altered_coeffs
    
    def integrated_wam(self, x,y):
        """
        implementation of the wam using integrated gradients
        """

        # compute the coeffs of the input
        if isinstance(x,list): # convert the input as tensor if necessary
            x=torch.tensor(
                np.array([wf/wf.max() for wf in x]).astype(np.float32)
            )


        # generate alpha with a given number of steps
        alphas=np.linspace(0,1,self.n_samples)


        # compute the coefficients 
        coeffs=ptwt.wavedec(x,self.wavelet, level=self.J,mode=self.mode)
        
        # compute z and melspec_z
        baseline_z=to_numpy(coeffs, 1, grads=False)
        baseline_melspec=self.compute_melspec(x,self.n_fft, self.sample_rate, self.n_mels)
        baseline_melspec=baseline_melspec.squeeze(1).detach().cpu().numpy()
        path_melspecs=[]
        path_grads=[]

        path_melspecs=np.empty((baseline_melspec.shape[0], 
                               self.n_samples, 
                               baseline_melspec.shape[1],
                               baseline_melspec.shape[2]))

        for i,alpha in enumerate(alphas):
             
            # compte alpha * z
            path_coeffs=self.alter(alpha,coeffs)

            path_melspec, path_grad=self.wam(path_coeffs,y,waveform=False)
            path_melspecs[:,i,:,:]=path_melspec
            path_grads.append(path_grad)

        # compute the integral for the melspec
        intergral_melspec=np.trapz(path_melspecs, axis=1)

        # for the gradient, integrate level wise each level
        integral_coeffs_tmp=[]

        for level in range(self.J+1):
            integral_coeffs_tmp.append(np.array(
                [pg[level] for pg in path_grads]
            ))

        integral_coeffs=[]
        for level_grads in integral_coeffs_tmp:
            
            integral_coeffs.append(
                np.trapz(level_grads, axis=0)
            )

        product_coefficients=[]
        for baseline_level, integrated_level in zip(baseline_z, integral_coeffs):
            product_coefficients.append(
                baseline_level * integrated_level
            )


        self.melspecs=baseline_melspec * intergral_melspec
        self.grad_coeffs=product_coefficients

        return baseline_melspec * intergral_melspec, product_coefficients

    def __call__(self,x,y):
        """
        calls n_samples times the BaseWAM1D on noisy samples
        stores the averaged gradients and the averaged melspecs
        """
        
        if self.method=="smooth":

            return self.smooth_wam(x,y)

        elif self.method=="integratedgrad":

            return self.integrated_wam(x,y)

        

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def process_in_chunks(melspec, chunk_size, sr, n_fft):
    spectrogram_chunks = []
    for i in range(0, melspec.shape[1], chunk_size):
        chunk = melspec[:, i:i + chunk_size]
        spectrogram_chunk = librosa.feature.inverse.mel_to_stft(chunk, sr=sr, n_fft=n_fft)
        spectrogram_chunks.append(spectrogram_chunk)
    return np.hstack(spectrogram_chunks)


class VisualizerWAM1D(WaveletAttribution1D):
    def __init__(self, model, x, wavelet="haar", J=3, method="smooth", mode="reflect", device=None, approx_coeffs=False, n_mels=128, n_fft=1024, sample_rate=44100, n_samples=25, stdev_spread=0.001, random_seed=42):
        super().__init__(model, wavelet, J, method, mode, device, approx_coeffs, n_mels, n_fft, sample_rate, n_samples, stdev_spread, random_seed)
        self.source_spectrograms=None
        self.x=x

    def compute_melspec(self,x):
        """computes the melspectrogram in
        power scale """

        if isinstance(x, list): 
            x=torch.tensor(
                np.array([wf/wf.max() for wf in x]).astype(np.float32)
            )

        mel_spectrogram=MelSpectrogram(sample_rate=self.sample_rate,
                                       n_fft=self.n_fft,n_mels=self.n_mels)

        melspectrograms=[]
        for waveform in x:
              melspec=mel_spectrogram(waveform)
              melspectrograms.append(
                  melspec.squeeze(-1)
                  )

        return torch.stack(melspectrograms).numpy().astype(np.float32)

    def compute_spectrogram(self,melspecs, chunk_size=100):
        """
        computes the spectrogram
        """

        spectrograms=[]
        for melspec in tqdm.tqdm(melspecs):
            spectrogram=process_in_chunks(melspec,chunk_size,self.sample_rate,self.n_fft)
            spectrograms.append(spectrogram)

        return np.array(spectrograms)
    
    def filter_melspec(self, audio_melspecs, grad_melspecs, filtering_method, EPS=0.2):
        """
        filters the melspectrogram of the waveform
        according to two approaches: 
        - hard thresholding (ht): applies a binary mask, with a threshold value EPS
        - modulation (modulation): computes melspec * grad_melspec directly

        
        args:
            - melspecs: the melspec of the waveforms
            - grad_melspec: the gradients on the melspec
            - filtering_mode: the type of filtering
            - EPS: the threshold, which shouln't be None if ht is chosen

        returns a filtered melspec
        """

        # transpose the gradients to fit the size of the melspecs
        grad_melspecs=np.transpose(grad_melspecs,(0,2,1))

        if filtering_method=="ht":

            # applies a threshold to the gradients
            grad_melspecs=normalize(grad_melspecs)
            mask = (grad_melspecs > EPS).astype(int)

            return audio_melspecs * mask
        
        elif filtering_method=="modulation":

            return audio_melspecs * np.abs(grad_melspecs)
        
    def spectrogram_from_waveform(self,waveform):

        # Compute the STFT (Spectrogram)

        hop_length=self.n_fft // 4 # for a good time resolution

        spectrograms = librosa.stft(waveform, n_fft=self.n_fft, hop_length=hop_length)

        return np.abs(spectrograms) # return the modulus of the spectrograms
    
    def filter_from_wavelet_coefficients(self, coefficients, gradients,
                                         filtering_method='ht',
                                         EPS=0.2):
        """
        filters the waveform in the wavelet domain and 
        return the filtered waveform.
        """

        # define the mask
        # filters the normalized absolute value of the gradient

        if filtering_method=="ht":
            masks=[
                (np.abs(grads) / grads.max() > EPS).astype(int) for grads in gradients
            ]

            filtered_coeffs=[
                coeff * mask for coeff, mask in zip(coefficients,masks)
            ]

        elif filtering_method=="st":

            # normalize the coefficients

            masks = [
                np.maximum(
                    normalize(coeff * grads) - EPS, 0
                )
                for coeff, grads in zip(coefficients, gradients)
            ]

            filtered_coeffs=[
                coeff * mask for coeff, mask in zip(coefficients,masks)
            ]
        elif filtering_method=="modulation":

            # sum the importance of each scale
            importances=np.array([
                np.sum(grads, axis=1) for grads in gradients
            ])

            normalized_importance=np.array([importances[:,i]/np.sum(importances[:,i]) for i in range(importances.shape[1])]).transpose()


            # modulate the coefficients by the gradients
            _filtered_coeffs=[
                (coeff * np.abs(grads)) for coeff, grads in zip(coefficients,gradients)
            ]

            filtered_coeffs=[]
            for coeff, importance in zip(_filtered_coeffs, normalized_importance):
                filtered_coeffs.append(coeff * importance[:,np.newaxis])

        filtered_sounds=pywt.waverec(filtered_coeffs, wavelet=self.wavelet)

        return filtered_sounds

    def filtered_spectrogram_from_wavelet_coefficients(self,
                                                       grad_coeffs,
                                                       filtering_method,
                                                       EPS=0.2,
                                                       ):
        """
        generates a filtered spectrogram by filtering 
        the input signal in the wavelet domain, according to different 
        filtering methods. See the docstring of the method
        that filters the wavelet coefficients for more details

        returns the source and the filtered spectrograms
        """
        # convert a the proper type
        if np.array(self.x).dtype == np.int16:
            self.x = np.array([wf/wf.max() for wf in self.x]).astype(np.float32)


        # compute the melspec
        self.source_spectrograms=self.spectrogram_from_waveform(self.x)

        # give the gradients and the wavelet transform of the 
        # waveform, compute the 
        coeffs=pywt.wavedec(self.x, self.wavelet,level=self.J,mode=self.mode)
        filtered_sounds=self.filter_from_wavelet_coefficients(coeffs,grad_coeffs,filtering_method=filtering_method,EPS=EPS)

        return self.source_spectrograms, self.spectrogram_from_waveform(filtered_sounds)

        #return self.source_spectrograms, self.spectrogram_from_waveform()

    def filtered_spectrogram_from_melspec(self,
                                          grad_melspecs,
                                          filtering_method,
                                          EPS=0.2,
                                          chunk_size=100
                                    ):
        """
        returns the filtered spectrogram of x 
        from the gradients computed on the melspectrogram
        
        returns source_spectrograms and filtered_spectrograms,
        arrays corresponding to the spectrograms before and after 
        filtering
        
        """

        # compute the melspec
        audio_melspecs=self.compute_melspec(self.x)
        self.source_spectrograms=self.compute_spectrogram(audio_melspecs,chunk_size=chunk_size)

        # filter the melspec
        filtered_melspecs=self.filter_melspec(audio_melspecs, grad_melspecs,filtering_method,EPS=EPS)

        return self.source_spectrograms, self.compute_spectrogram(filtered_melspecs,chunk_size=chunk_size)

