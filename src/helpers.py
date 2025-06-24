# -*- coding: utf-8 -*-

# libraries
import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import lib.network_architectures as netark
import pandas as pd
import os
import torch.nn.functional as F
import json
import scipy.io.wavfile as wavf
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import colorsys
#import h5py
from lib.network_architectures import PointNetCls, VoxelModel


netType = getattr(netark, 'weak_mxh64_1024')
netwrkgpl = F.max_pool2d # keep it fixed. It was avg_pool2d for experiments on ESC, initial experiments on SONYC_UST

def normalize_data(data):
    return(data - np.min(data)) / (np.max(data)-np.min(data))


def add_0db_noise(audio):
    """
    Add Gaussian noise to an audio signal such that the noise has 0 dB SNR (i.e., same power as the input signal).

    Parameters:
    audio (numpy array): The input audio signal (in int16 format).

    Returns:
    noisy_audio (numpy array): The noisy audio signal with 0 dB noise and same int16 datatype.
    """
    
    # Convert the audio to float32 for safe computation
    audio_float = audio.astype(np.float32)

    # Calculate the RMS (Root Mean Square) of the audio signal
    rms_signal = np.sqrt(np.mean(audio_float ** 2))

    # Generate Gaussian noise
    noise = np.random.normal(0, 1, audio_float.shape)

    # Calculate the RMS of the noise
    rms_noise = np.sqrt(np.mean(noise ** 2))

    # Scale the noise to have the same RMS as the signal
    noise = noise * (rms_signal / rms_noise)

    # Add noise to the audio signal
    noisy_audio_float = audio_float + noise

    # Clip the values to avoid overflow/underflow, ensuring they remain in the int16 range
    noisy_audio_clipped = np.clip(noisy_audio_float, -32768, 32767)

    # Convert back to int16
    noisy_audio = noisy_audio_clipped.astype(np.int16)

    return noisy_audio


def filter_coeffs(coeffs, EPS, normalized=False):
    """filters the input coeff"""

    if normalized:
        return (coeffs >= EPS).astype(int)
    else:
        normalized_coeffs= (coeffs-coeffs.min()) / (coeffs.max() - coeffs.min())
        
        return (normalized_coeffs > EPS).astype(int)


def load_3d_model(source_dir,num_classes,feature_transform):
    """
    loads the point net model
    """

    model_dir=os.path.join(source_dir,'model-weights')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load pre-trained weights
    model = PointNetCls(k=num_classes, feature_transform=feature_transform)
    model.load_state_dict(torch.load(os.path.join(model_dir,'pointNet.pth')))
    model.eval().to(device)

    return model

def load_3dVoxel_model(source_dir,num_classes):
    """
    loads the Voxel net model
    """

    model_dir=os.path.join(source_dir,'model-weights')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load pre-trained weights
    model = VoxelModel(n_out_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir,'VoxelModel.pth')))
    model.eval().to(device)

    return model

def load_3d_mnist(source_dir, batch_size, num_points=1024, train=False, seed=42):
    """
    loads the 3D MNIST from the source directory. 

    It is assumed that the dataset is in a folder "3DMNIST", as 
    in the attached data file.
    
    If train is false, returns only the test instances
    """

    # directory to the data
    data_dir=os.path.join(source_dir,"3DMNIST")

    # Fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_file=os.path.join(data_dir,"test_point_clouds.h5")
    train_file=os.path.join(data_dir,"train_point_clouds.h5")

    # Test Set
    with h5py.File(test_file, 'r') as points_dataset:
        X_test = []
        targets_test = []
        for i in range(len(points_dataset)):
            pc = points_dataset[str(i)]['points'][:]
            idx = np.random.choice(pc.shape[0], num_points)
            X_test.append(pc[idx])
            targets_test.append(points_dataset[str(i)].attrs['label'])
    X_test = np.array(X_test)
    targets_test = np.array(targets_test)

    test_x = torch.from_numpy(X_test).float()
    test_y = torch.from_numpy(targets_test).long()

    test_ds = torch.utils.data.TensorDataset(test_x,test_y)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    if train: # add the train instances if necessary

        with h5py.File(train_file, 'r') as points_dataset:
            X_train = []
            targets_train= []
            for i in range(len(points_dataset)):
                pc = points_dataset[str(i)]['points'][:]
                # Randomly select NUM_POINTS points 
                idx = np.random.choice(pc.shape[0], num_points)
                X_train.append(pc[idx])
                targets_train.append(points_dataset[str(i)].attrs['label'])
        X_train = np.array(X_train)
        targets_train = np.array(targets_train)

        train_x = torch.from_numpy(X_train).float()
        train_y = torch.from_numpy(targets_train).long()

        # Initialize Datasets and DataLoaders
        train_ds = torch.utils.data.TensorDataset(train_x,train_y)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size = batch_size, shuffle = False)

        return (test_loader, test_ds), (train_loader, train_ds)
    else:
        return (test_loader, test_ds)
    
def load_3dVoxel_mnist(source_dir, batch_size, seed=42):
    """
    loads the 3D Voxel MNIST from the source directory. 

    It is assumed that the dataset is in a folder "3DMNIST", as 
    in the attached data file.
    
    """

    # directory to the data
    data_dir=os.path.join(source_dir,"3DMNIST")

    # Fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    with h5py.File(os.path.join(data_dir, "full_dataset_vectors.h5"), "r") as hf:    
        X_train = hf["X_train"][:]
        targets_train = hf["y_train"][:]
        X_test = hf["X_test"][:] 
        targets_test = hf["y_test"][:]


    # reshape voxels
    X_train = X_train.reshape(-1, 16, 16, 16)
    X_test = X_test.reshape(-1, 16, 16, 16)

    train_x = torch.from_numpy(X_train).float()
    print('train x shape:', train_x.shape)
    train_y = torch.from_numpy(targets_train).long()
    test_x = torch.from_numpy(X_test).float()
    test_y = torch.from_numpy(targets_test).long()

    batch_size = 32 # set a minimal batch size

    train_ds = torch.utils.data.TensorDataset(train_x,train_y)
    test_ds = torch.utils.data.TensorDataset(test_x,test_y)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True) 
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    return (test_loader, test_ds), (train_loader, train_ds)


def load_sound(root_dir, n=42, noise=False):
    """
    helper that returns a dictionnary with keys 'x' and 'y' corresponding
    to the soundwaves and their labels.

    extracted from the ECS50 dataset. Files formatting should be as in the 
    attached data folder.
    """
    df_sounds=pd.read_csv(os.path.join(
        root_dir,"meta/esc50.csv"
    ), sep=","
    )

    # load the sounds
    waveforms, labels=[], []

    if isinstance(n, list):
        indices=n

        for index in indices:
           
           wav_file_name=os.path.join(os.path.join(root_dir,"audio"), index)
           fs, inp_audio = wavf.read(wav_file_name) # returns the bitrate and the data from the file. Corresponds to a time series
           label=df_sounds[df_sounds["filename"]==index]['target'].item()
           labels.append(label)

           if noise:
               waveforms.append(add_0db_noise(inp_audio))
           else:
               waveforms.append(inp_audio)
    
    else:
    
        np.random.seed(42)
        indices=np.random.randint(0,df_sounds.shape[0],n)
    
        for index in indices:
            wav_file_name=os.path.join(os.path.join(root_dir,"audio"),df_sounds.loc[index,'filename'])
            fs, inp_audio = wavf.read(wav_file_name) # returns the bitrate and the data from the file. Corresponds to a time series 
            label=df_sounds.loc[index,'target']
            labels.append(label)

            if noise:
                waveforms.append(
                    add_0db_noise(inp_audio)
                )
            else:
                waveforms.append(inp_audio)

    return {'x' : waveforms, 'y' : labels}

def load_audio_model(root_model_dir, device="cpu"):
    """
    loads a model from L2I and returns it
    """

    f=FtEx(n_classes=50)
    f.load_state_dict(
        torch.load(os.path.join(root_model_dir,'best_full_model_Fold1_K100.pt'))['f_state_dict'])

    f.to(device).eval()

    return f

# Taken from L2I, class to load the corresponding model
class FtEx(nn.Module):
    def __init__(self, n_classes=10, multi_label=False):
        super(FtEx, self).__init__()
        self.netx = netType(527, netwrkgpl) # Only initially to load model
        #self.load_model(pre_model_path)
        self.layer = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU())
        self.fc = nn.Linear(256, n_classes, bias=True)
        self.reg = nn.Dropout(0.2)

    def forward(self, inp):
        out, out_inter = self.netx(inp)
        out = self.layer(out_inter[0])
        out = torch.flatten( netwrkgpl(out, kernel_size=out.shape[2:]), 1)
        out = self.reg(out)
        out = self.fc(out)
        return out#, out_inter
    
    def full_forward(self,inp):
        out, out_inter = self.netx(inp)
        out = self.layer(out_inter[0])
        out = torch.flatten( netwrkgpl(out, kernel_size=out.shape[2:]), 1)
        out = self.reg(out)
        out = self.fc(out)
        return out, out_inter

    def load_model(self, modpath):
        #load through cpu -- safest
        state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        self.netx.load_state_dict(new_state_dict)


def load_imagenet_validation(source_dir, 
                             ground_truth="val.txt", 
                             count=1000, 
                             seed=42,
                             transform=None):
    """
    loads the imagenet dataset. returns x and y where x are the images
    """

    if transform is None:
        # set the official imagenet transforms
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    # set the seed
    np.random.seed(seed)

    # open the ground truth file, assumed to be 
    # a txt file with a IMAGE_NAME \t LABEL_INDEX structure
    f=open(os.path.join(source_dir,ground_truth), 'r')
    ground_truth={
        line.strip().split()[0] : int(line.strip().split()[1]) for line in f 
    }

    # randomly pick count values or load the images of the folder
    # in this case, make sure that the folder contains only images img_name.JPEG and
    # the validation file val.txt
    # examples=np.random.choice(list(ground_truth.keys()), size=count, replace=False)
    examples=[e for e in os.listdir(source_dir) if e[-5:]==".JPEG"]
    assert len(examples)==count

    # load the images
    images=[Image.open(os.path.join(source_dir,example)).convert('RGB') for example in examples]
    y=[ground_truth[example] for example in examples]


    return torch.stack([transform(im) for im in images]), y

def load_images(source_dir=None,label_file="labels.json", labels=None, images_dir=None):
    """
    util to load the images, returns 
    the stacked tensor of images

    by default, assumes the same structure as in the assets file directory
    attached with the repository
    """

    if images_dir is not None:
        PREPROCESS=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        # Load the images
        PREPROCESS=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if labels is None:

        images_dir=os.path.join(source_dir,"assets")

        # preprocess a batch of images
        # compute the explanations
        # load the labels
        labels=json.load(open(os.path.join(images_dir,label_file)))

        names=list(labels.keys())
        labels_list=list(labels.values())

    else:
        names=os.listdir(images_dir)
        labels_list=labels

    images=[]
    for name in names:
        dest=os.path.join(images_dir, name)

        images.append(
            Image.open(dest).convert('RGB')
        )

    return torch.stack([PREPROCESS(im) for im in images]), labels_list

def show(img, p=False, inverse_c = False, smooth=1.2, plot=True,**kwargs):
  """ Display torch/tf tensor """
  img = np.array(img, dtype=np.float32)

  # check if channel first
  if img.shape[0] == 1:
    img = img[0]
  elif img.shape[0] == 3:
    img = np.moveaxis(img, 0, 2)
  # check if cmap
  if img.shape[-1] == 1:
    img = img[:,:,0]
  # normalize
  if img.max() > 1 or img.min() < 0:
    img -= img.min(); img/=img.max()
  # check if clip percentile
  if p is not False:
    img = np.clip(img, np.percentile(img, p), np.percentile(img, 100-p))

  if img.shape[-1] == 3 and inverse_c:
    img = img[...,::-1]

  if plot:
    plt.imshow(img, **kwargs)
    plt.axis('off')
    plt.grid(None)
  else:
    return img

def get_alpha_cmap(cmap, min_alpha=0.):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        c = np.array((cmap[0]/255.0, cmap[1]/255.0, cmap[2]/255.0))
        cmax = colorsys.rgb_to_hls(*c)
        cmax = np.array(cmax)
        cmax[-1] = 1.0
        cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c,cmax])

        alpha_cmap = cmap(np.arange(256))
        alpha_cmap[:,-1] = np.linspace(min_alpha, 0.85, 256)
        alpha_cmap = ListedColormap(alpha_cmap)

        return alpha_cmap


def load_vision_model(model_key="resnet18",checkpoint_path=None, device="cpu"):
   
    """
    loads a vision model from timm. Can specify the path to custom weights if necessary
    """

    if checkpoint_path is None:
        model=timm.create_model(model_key, pretrained=True)
    else:
        model=timm.create_model(model_key,checkpoint_path=checkpoint_path)

    return model.eval().to(device)