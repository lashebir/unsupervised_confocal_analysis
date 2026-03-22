#!/usr/bin/env python
# coding: utf-8

# # 00. Terminal Commands

# # Terminal Commands
# 
# `$token = Get-Content ".keys\gh_pt"`
# 
# 
# `git remote set-url origin "https://$token@github.com/lashebir/unsupervised_confocal_analysis.git"`
# 
# `jupyter nbconvert --to script shallow_cnn_learning.ipynb`
# 
# 
# 
# `Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-u shallow_cnn_learning_updated.py" -RedirectStandardOutput "..\finetuning\train_log.txt" -RedirectStandardError "..\finetuning\train_error.txt"`

# # 0. Imports & Functions

# ## Imports

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

from aicsimageio.readers.lif_reader import LifReader
from readlif.reader import LifFile
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch.nn.functional as F
# import skfda, L2Regularization, LinearDifferentialOperator, BSplineBasis, BasisSmoother   # FOR WAVEFORM DATA

# from ABRA_022626 import interpolate_and_smooth, CNN, plot_wave, calculate_and_plot_wave, plot_waves_single_frequency, arfread, get_str, calculate_hearing_threshold, all_thresholds, peak_finding
from sklearn.manifold import TSNE
import pandas as pd
import sys
import zarr
import numpy as np
import time

import re
import io

import os
import torch
import aicsimageio

from scipy.optimize import curve_fit
# from skfda.representation.basis import BSplineBasis
# from skfda.misc.regularization import L2Regularization
# from skfda.misc.operators import LinearDifferentialOperator
# from skfda.preprocessing.smoothing import BasisSmoother
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans, HDBSCAN
# from collections import defaultdict
# import skfda

from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, GroupKFold


import datetime
# from skfda import FDataGrid
# from skfda.preprocessing.dim_reduction import FPCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import torch.nn as nn
# from tensorflow.keras.models import load_model
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import random
# import tensorflow as tf
import torch.autograd as autograd
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
import torchvision.transforms as transforms

import pacmap
import pickle


# ## ABRA

# In[ ]:


class CNN(nn.Module):
    def __init__(self, filter1, filter2, dropout1, dropout2, dropout_fc):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filter1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=filter1, out_channels=filter2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(filter2 * 61, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.batch_norm1 = nn.BatchNorm1d(filter1)
        self.batch_norm2 = nn.BatchNorm1d(filter2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(-1, self.fc1.in_features)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


def interpolate_and_smooth(final, target_length=244):
    if len(final) > target_length:
        new_points = np.linspace(0, len(final), target_length + 2)
        interpolated_values = np.interp(new_points, np.arange(len(final)), final)
        final = np.array(interpolated_values[:target_length], dtype=float)
        final = pd.Series(final)
    elif len(final) < target_length:
        original_indices = np.arange(len(final))
        target_indices = np.linspace(0, len(final) - 1, target_length)
        cs = CubicSpline(original_indices, final)
        final = cs(target_indices)
    return final


# In[ ]:


def latency_all_peaks(highest_peaks, y_values, time_scale):
    latencies = []
    num_peaks = highest_peaks.size
    if num_peaks > 0:  # Check if highest_peaks is not empty
        for n in range(num_peaks): # SHOULD be 5 but there are cases where there are less. Will handle in later loops
            lat = highest_peaks[n] * (time_scale / len(y_values)) # Based on ABRA logic
            latencies.append(lat)
        return latencies
    else:
        print("No peaks detected. Check input data")
        return None


# In[ ]:


def full_interpolation(df, freq, db, time_scale=10, multiply_y_factor=1.0, units='Microvolts'):

    khz = df[(df['Freq(kHz)'] == freq) & (df['Level(dB)'] == db)]
    # print(khz)
    if not khz.empty:
        index = khz.index.values[0]
        final = df.loc[index, '0':].dropna()
        final = pd.to_numeric(final, errors='coerce').dropna()

        target = int(244 * (time_scale / 10))

        # Process the wave as in calculate_and_plot_wave
        y_values = interpolate_and_smooth(final, target)

        # print(f"Interpolated y_values: {y_values[:5]}")
        # print(f"Any NaNs? {np.isnan(y_values).any()}")

        if final.empty:
            print(f"Warning: Empty waveform for {freq}kHz @ {db}dB")
            return np.full((1, 244), np.nan)

        # Apply scaling factor
        y_values *= multiply_y_factor

        # Handle units conversion if needed
        if units == 'Nanovolts':
            y_values /= 1000

        # Generate normalized version for peak finding
        y_values_fpf = interpolate_and_smooth(y_values[:244])

        # Standardize and normalize for peak finding, exactly as in the original
        flattened_data = y_values_fpf.flatten().reshape(-1, 1)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(flattened_data)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = min_max_scaler.fit_transform(standardized_data).reshape(y_values_fpf.shape)

        return scaled_data


# In[ ]:


def peak_finding(wave):
    # Prepare waveform
    waveform=interpolate_and_smooth(wave) # Added indexing per calculate and plot wave function
    # waveform_torch = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0) archived ABRA
    waveform_torch = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0) #newer ABRA
    # print(waveform_torch)
    # Get prediction from model
    outputs = peak_finding_model(waveform_torch)
    prediction = int(round(outputs.detach().numpy()[0][0], 0))
    # prediction_test = int(round(outputs.detach().numpy()[0], 0))
    # print("Model output:", outputs, "Prediction true start:", prediction)

    # Apply Gaussian smoothing
    smoothed_waveform = gaussian_filter1d(waveform, sigma=1)

    # Find peaks and troughs
    n = 18
    t = 14
    # start_point = prediction - 9 archived ABRA
    start_point = prediction - 6 #newer ABRA
    smoothed_peaks, _ = find_peaks(smoothed_waveform[start_point:], distance=n)
    smoothed_troughs, _ = find_peaks(-smoothed_waveform, distance=t)
    sorted_indices = np.argsort(smoothed_waveform[smoothed_peaks+start_point])
    highest_smoothed_peaks = np.sort(smoothed_peaks[sorted_indices[-5:]] + start_point)
    relevant_troughs = np.array([])
    for p in range(len(highest_smoothed_peaks)):
        c = 0
        for t in smoothed_troughs:
            if t > highest_smoothed_peaks[p]:
                if p != 4:
                    try:
                        if t < highest_smoothed_peaks[p+1]:
                            relevant_troughs = np.append(relevant_troughs, int(t))
                            break
                    except IndexError:
                        pass
                else:
                    relevant_troughs = np.append(relevant_troughs, int(t))
                    break
    relevant_troughs = relevant_troughs.astype('i')
    return highest_smoothed_peaks, relevant_troughs

def extract_metadata(metadata_lines):
    # Dictionary to store extracted metadata
    metadata = {}

    for line in metadata_lines:
        # Extract SW FREQ
        freq_match = re.search(r'SW FREQ:\s*(\d+\.?\d*)', line)
        if freq_match:
            metadata['SW_FREQ'] = float(freq_match.group(1))

        # Extract LEVELS
        levels_match = re.search(r':LEVELS:\s*([^:]+)', line)
        if levels_match:
            # Split levels and convert to list of floats
            metadata['LEVELS'] = [float(level) for level in levels_match.group(1).split(';') if level]

    return metadata

def read_custom_tsv(file_path):
    # Read the entire file
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        content = f.read()

    # Split the content into metadata and data sections
    metadata_lines = []
    data_section = None

    # Find the ':DATA' marker
    data_start = content.find(':DATA')

    if data_start != -1:
        # Extract metadata (lines before ':DATA')
        metadata_lines = content[:data_start].split('\n')

        # Extract data section
        data_section = content[data_start:].split(':DATA')[1].strip()

    # Extract specific metadata
    metadata = extract_metadata(metadata_lines)

    # Read the data section directly
    try:
        # Use StringIO to create a file-like object from the data section
        raw_data = pd.read_csv(
            io.StringIO(data_section), 
            sep='\s+',  # Use whitespace as separator
            header=None
        )
        raw_data = raw_data.T
        # Add metadata columns to the DataFrame
        if 'SW_FREQ' in metadata:
            raw_data['Freq(kHz)'] = metadata['SW_FREQ']
            # raw_data['Freq(Hz)'] = raw_data['Freq(Hz)'].apply(lambda x: x*1000)

        if 'LEVELS' in metadata:
            # Repeat levels to match the number of rows
            levels_repeated = metadata['LEVELS'] * (len(raw_data) // len(metadata['LEVELS']) + 1)
            raw_data['Level(dB)'] = levels_repeated[:len(raw_data)]

        filtered_data = raw_data.apply(pd.to_numeric, errors='coerce').dropna()
        filtered_data.columns = filtered_data.columns.map(str)

        columns = ['Freq(kHz)'] + ['Level(dB)'] + [col for col in filtered_data.columns if col.isnumeric() == True]
        filtered_data = filtered_data[columns]
        return filtered_data

    except Exception as e:
        print(f"Error reading data: {e}")
        return None, metadata


# In[ ]:


def peaks_troughs_amp_final(df, freq, db, time_scale=10, multiply_y_factor=1.0, units='Microvolts'):
    db_column = 'Level(dB)'

    khz = df[(df['Freq(kHz)'] == freq) & (df[db_column] == db)]
    if not khz.empty:
        index = khz.index.values[0]
        final = df.loc[index, '0':].dropna()
        final = pd.to_numeric(final, errors='coerce').dropna()

        target = int(244 * (time_scale / 10))

        # Process the wave as in calculate_and_plot_wave
        y_values = interpolate_and_smooth(final, target)

        # Apply scaling factor
        y_values *= multiply_y_factor

        # Handle units conversion if needed
        if units == 'Nanovolts':
            y_values /= 1000

        # Generate normalized version for peak finding
        y_values_fpf = interpolate_and_smooth(y_values[:244])

        # Standardize and normalize for peak finding, exactly as in the original
        flattened_data = y_values_fpf.flatten().reshape(-1, 1)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(flattened_data)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = min_max_scaler.fit_transform(standardized_data).reshape(y_values_fpf.shape)
        y_values_fpf = interpolate_and_smooth(scaled_data[:244])

        # Find peaks using the normalized data
        highest_peaks, relevant_troughs = peak_finding(y_values_fpf)

        # Calculate amplitude on the processed but non-normalized data
        if highest_peaks.size > 0 and relevant_troughs.size > 0:
            # Following the same approach as in the display_metrics_table function
            first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
            return highest_peaks, relevant_troughs, first_peak_amplitude

    return None, None, None


# ## Data Loading

# In[ ]:


def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data_norm.max()
    data_norm = data_norm * scale_fact
    return data_norm.astype(dtype)

def convert_max_proj_tensor(file_path, ch_order=['myo7', 'glur2', 'ctbp2', 'nf']):
    try:
        data = np.load(file_path)  # (C, Z, H, W)

        all_channels = []
        for ch in range(data.shape[0]):
            ch_name = ch_order[ch]
            channel_data = data[ch]  # (Z, H, W)

            if ch_name == 'ctbp2':
                channel_data = normalize(channel_data.astype(np.uint16), maxval=(2**16 - 1)).astype(np.uint16)

            max_proj = np.max(channel_data, axis=0)  # (H, W)
            all_channels.append(max_proj)

        image_3d = np.stack(all_channels, axis=0)  # (C, H, W)
        tensor = torch.from_numpy(image_3d).float()
        return tensor

    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")
        return None


def normalize_per_channel(tensor):
    """
    Normalize each channel independently to [0, 1].

    Args:
        tensor: Shape [C, H, W]

    Returns:
        Normalized tensor of same shape
    """
    normalized = torch.zeros_like(tensor)

    for c in range(tensor.shape[0]):
        channel = tensor[c]
        normalized[c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

    return normalized


# In[ ]:


class SynapseImageDataset(Dataset):
    """
    Dataset that handles v1/v2 pairing from filenames.
    Skips files that cannot be loaded.
    """
    def __init__(self, image_paths, ch_order=['myo7', 'glur2', 'ctbp2', 'nf'], 
                 target_size=224, cache_path = r'D:\Leah\Liberman Data\preloaded_cache.pt'):
        self.ch_order = ch_order
        self.target_size = target_size
        self.resize = transforms.Resize((target_size, target_size))

        # Build pairing dictionary
        self.pairs = {}  # Maps each file to its pair
        self.image_paths = []

        self.cache = {}  # Cache for loaded tensors in memory instead of from disk every epoch

        for path in image_paths:

            if not path.endswith('.npy'):
                continue

            self.image_paths.append(path)

            # Extract base name and version
            if '.v1.npy' in path:
                base = path.replace('.v1.npy', '')
                pair_path = base + '.v2.npy'

            elif '.v2.npy' in path:
                base = path.replace('.v2.npy', '')
                pair_path = base + '.v1.npy'

            else:
                # No version in filename, skip pairing
                continue

            # Store the pairing
            self.pairs[path] = pair_path

        self.path_to_idx = {path: idx for idx, path in enumerate(self.image_paths)}  # Map paths to indices for quick lookup

        # # Preload everything into memory
        # print(f"Preloading {len(self.image_paths)} images into RAM...")
        # failed = 0
        # for idx, path in enumerate(self.image_paths):
        #     tensor = convert_max_proj_tensor(path, self.ch_order)
        #     if tensor is not None:
        #         tensor = normalize_per_channel(tensor)
        #         tensor = self.resize(tensor.unsqueeze(0)).squeeze(0)
        #         self.cache[idx] = tensor
        #     else:
        #         failed += 1
        #     if (idx + 1) % 100 == 0:
        #         print(f"  Loaded {idx + 1}/{len(self.image_paths)}")

        # After preloading, save the cache to disk
        # torch.save(self.cache, r'D:\Leah\preloaded_cache.pt')

        # Save the preloaded cache to disk for future use. ONLY NEED TO CHANGE IF THE FOLLOWING CHANGE:
        # target_size, ch_order, normalization logic, adding new images to the dataset

        # print(f"Cached: {len(self.cache)}, Failed: {failed}")

        # NEW APPROACH: preload when not available, otherwise use existing file

        # Try loading from saved cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading preloaded cache from {cache_path}...")
            self.cache = torch.load(cache_path)
            print(f"Loaded {len(self.cache)} cached tensors")
        else:
            # Preload everything into memory
            print(f"No existing cache detected. Preloading {len(self.image_paths)} images into RAM...")
            failed = 0
            for idx, path in enumerate(self.image_paths):
                tensor = convert_max_proj_tensor(path, self.ch_order)
                if tensor is not None:
                    tensor = normalize_per_channel(tensor)
                    tensor = self.resize(tensor.unsqueeze(0)).squeeze(0)
                    self.cache[idx] = tensor
                else:
                    failed += 1
                if (idx + 1) % 100 == 0:
                    print(f"  Loaded {idx + 1}/{len(self.image_paths)}")

            print(f"Cached: {len(self.cache)}, Failed: {failed}")

            # Save for future use
            if cache_path:
                torch.save(self.cache, cache_path)
                print(f"Saved cache to {cache_path}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns: Single tensor [4, H, W]
        Returns None if file cannot be loaded (error already printed by convert_max_proj_tensor)
        """
        # img_path = self.image_paths[idx]
        # tensor = convert_max_proj_tensor(img_path, self.ch_order)

        # # If loading failed, return None
        # if tensor is None:
        #     return None

        # tensor = normalize_per_channel(tensor)
        # tensor = self.resize(tensor.unsqueeze(0)).squeeze(0)
        # return tensor

        if idx in self.cache:
            return self.cache[idx]
        return None

    # def get_pair(self, idx):
    #     """
    #     Get the paired image for index idx.
    #     Returns (image, pair_image) or (None, None) if loading fails.
    #     If pair doesn't exist or fails to load, returns (image, None).
    #     """
    #     # Get original image
    #     img_path = self.image_paths[idx]
    #     img = self[idx]

    #     # If original image failed to load, return (None, None)
    #     if img is None:
    #         print(f"  Skipping pair lookup - original image failed to load")
    #         return None, None

    #     # Check if pair exists
    #     if img_path not in self.pairs:
    #         return img, None

    #     pair_path = self.pairs[img_path]

    #     # Load pair image
    #     # if pair_path in self.image_paths:
    #     #     pair_idx = self.image_paths.index(pair_path)
    #     #     pair_img = self[pair_idx]

    #     #Load pair image from cache if available
    #     if pair_path in self.path_to_idx:
    #         pair_idx = self.path_to_idx[pair_path]
    #         pair_img = self[pair_idx]

    #         if pair_img is None:
    #             print(f"  Pair image failed to load: {pair_path}")
    #             return img, None

    #         return img, pair_img
    #     else:
    #         return img, None


    def get_all_valid_pairs(self):
        """Pre-compute all valid pair indices"""
        valid_pairs = []
        for idx in range(len(self.image_paths)):
            path = self.image_paths[idx]
            if path in self.pairs and self.pairs[path] in self.path_to_idx:
                pair_idx = self.path_to_idx[self.pairs[path]]
                if idx in self.cache and pair_idx in self.cache:
                    valid_pairs.append((idx, pair_idx))
        return valid_pairs


def collate_fn_skip_none(batch):
    """
    Custom collate function that filters out None values from batches.
    This allows the DataLoader to skip corrupted images that failed to load.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]

    # If all items were None, return empty batch
    if len(batch) == 0:
        return torch.empty(0)

    # Stack remaining valid tensors
    return torch.stack(batch, 0)


def create_dataloader(image_paths, batch_size=8, shuffle=True, augment=False, 
                      num_workers=4, target_size=224):
    """
    Create PyTorch DataLoader for synapse images.

    Args:
        image_paths: List of .lif file paths
        batch_size: Batch size for training
        shuffle: Shuffle data
        augment: Apply augmentations (for Shallow CNN training)
        num_workers: Number of parallel workers
        target_size: Image size

    Returns:
        DataLoader object
    """
    dataset = SynapseImageDataset(
        image_paths=image_paths,
        target_size=target_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn_skip_none  # Custom collate to skip None values
    )

    return dataloader


# def collect_all_lif_paths(data_dir, subjects=None):
#     """
#     Collect all .lif file paths from directory structure.

#     Args:
#         data_dir: Base directory (e.g., liberman_data/Confocal Data Charles Liberman/)
#         subjects: Optional list of subject IDs to include (e.g., ['WPZ116', 'WPZ145'])
#                   If None, include all subjects

#     Returns:
#         List of full paths to .lif files
#     """
#     lif_paths = []

#     for subject_dir in os.listdir(data_dir):
#         if re.match(r'WPZ\d+', subject_dir):  # filtering out folders like "WPZ 104"
#             subject_path = os.path.join(data_dir, subject_dir)

#             # Filter by subjects if specified
#             if subjects is not None and subject_dir not in subjects:
#                 continue

#             if os.path.isdir(subject_path):
#                 for file_name in os.listdir(subject_path):
#                     if file_name.endswith('.lif'):
#                         full_path = os.path.join(subject_path, file_name)
#                         lif_paths.append(full_path)

#     print(f"Found {len(lif_paths)} .lif files")
#     return lif_paths


def collect_all_npy_paths(data_dir, subjects=None):
    npy_paths = []

    for subject_dir in os.listdir(data_dir):
        if re.match(r'WPZ\d+$', subject_dir):
            subject_path = os.path.join(data_dir, subject_dir)

            if subjects is not None and subject_dir not in subjects:
                continue

            if os.path.isdir(subject_path):
                for file_name in os.listdir(subject_path):
                    if file_name.endswith('.npy'):
                        full_path = os.path.join(subject_path, file_name)
                        npy_paths.append(full_path)

    print(f"Found {len(npy_paths)} .npy files")
    return npy_paths


# ## Clustering

# In[ ]:


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def cluster_embeddings(embeddings, method='kmeans', n_clusters=5, **kwargs):
    """
    Cluster embeddings using various algorithms.

    Args:
        embeddings: numpy array of shape [n_samples, embedding_dim]
        method: 'kmeans', 'dbscan', or 'hierarchical'
        n_clusters: Number of clusters (for kmeans/hierarchical)
        **kwargs: Additional arguments for clustering algorithm

    Returns:
        cluster_labels: Array of cluster assignments
        cluster_model: Fitted clustering model
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = model.fit_predict(embeddings)

    # elif method == 'dbscan':
    #     from sklearn.cluster import DBSCAN
    #     eps = kwargs.get('eps', 0.5)
    #     min_samples = kwargs.get('min_samples', 5)
    #     model = DBSCAN(eps=eps, min_samples=min_samples)
    #     labels = model.fit_predict(embeddings)

    elif method == 'hdbscan':
        min_cluster_size = kwargs.get('min_cluster_size', 10)
        min_samples = kwargs.get('min_samples', 5)
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = model.fit_predict(embeddings)

    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        linkage = kwargs.get('linkage', 'ward')
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate silhouette score if we have multiple clusters
    n_unique = len(np.unique(labels))
    if n_unique > 1 and n_unique < len(embeddings):
        sil_score = silhouette_score(embeddings, labels)
        print(f"Silhouette Score: {sil_score:.3f}")
    else:
        print(f"Warning: Only {n_unique} unique cluster(s) found")

    return labels


def find_optimal_clusters(embeddings, max_k=10, method='kmeans'):
    """
    Find optimal number of clusters using elbow method and silhouette score.

    Args:
        embeddings: numpy array of embeddings
        max_k: Maximum number of clusters to try
        method: Clustering method

    Returns:
        Dictionary with scores for each k
    """
    results = {
        'k_values': [],
        'inertias': [],
        'silhouette_scores': []
    }

    for k in range(2, max_k + 1):
        labels, model = cluster_embeddings(embeddings, method=method, n_clusters=k)

        results['k_values'].append(k)

        if hasattr(model, 'inertia_'):
            results['inertias'].append(model.inertia_)

        if len(np.unique(labels)) > 1:
            sil = silhouette_score(embeddings, labels)
            results['silhouette_scores'].append(sil)
        else:
            results['silhouette_scores'].append(0)

        print(f"k={k}: Silhouette={results['silhouette_scores'][-1]:.3f}")

    return results


def analyze_clusters(embeddings, labels, metadata_df=None, metadata_cols=None):
    """
    Analyze cluster composition.

    Args:
        embeddings: numpy array of embeddings
        labels: Cluster labels
        metadata_df: DataFrame with metadata (subjects, freqs, groups, etc.)
        metadata_cols: List of column names to analyze

    Returns:
        Dictionary with cluster statistics
    """
    n_clusters = len(np.unique(labels))
    print(f"\n{'='*60}")
    print(f"CLUSTER ANALYSIS: {n_clusters} clusters found")
    print(f"{'='*60}\n")

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        n_samples = np.sum(mask)

        print(f"Cluster {cluster_id}: {n_samples} samples ({100*n_samples/len(labels):.1f}%)")

        if metadata_df is not None and metadata_cols is not None:
            cluster_metadata = metadata_df[mask]

            for col in metadata_cols:
                if col in cluster_metadata.columns:
                    value_counts = cluster_metadata[col].value_counts()
                    print(f"  {col}: {dict(value_counts)}")

        print()

    return labels


def compare_clusters_to_experimental_groups(labels, group_labels, group_names=None):
    """
    Compare discovered clusters to known experimental groups.

    Args:
        labels: Cluster assignments
        group_labels: True experimental group labels
        group_names: Optional names for groups

    Returns:
        Confusion matrix and purity scores
    """
    from sklearn.metrics import confusion_matrix, adjusted_rand_score

    # Confusion matrix
    cm = confusion_matrix(group_labels, labels)

    # Adjusted Rand Index (similarity between clusterings)
    ari = adjusted_rand_score(group_labels, labels)

    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"(1.0 = perfect match, 0.0 = random, <0 = worse than random)")

    print(f"\nConfusion Matrix:")
    print(cm)

    return cm, ari


# ## Cluster Visualization

# In[ ]:


def visualize_embeddings_2d(embeddings, labels=None, method='pacmap', title='Embedding Visualization',
                            colors=None, figsize=(10, 8), save_path=None):
    """
    Visualize high-dimensional embeddings in 2D.

    Args:
        embeddings: numpy array of shape [n_samples, embedding_dim]
        labels: Optional cluster labels or experimental groups
        method: 'pacmap' or 'tsne'
        title: Plot title
        colors: Optional custom colors for each point
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        2D embedding coordinates
    """
    print(f"Reducing {embeddings.shape[1]}D embeddings to 2D using {method.upper()}...")

    if method == 'pacmap':
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    plt.figure(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                       c=[cmap(i)], label=f'Cluster {label}', 
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        plt.legend()
    else:
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                   c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return embedding_2d


def plot_training_curves(losses, model_name='Model'):
    """
    Plot training loss curves.

    Args:
        losses: Loss dictionary or list from training
        model_name: Name for plot title
    """
    plt.figure(figsize=(10, 5))

    if isinstance(losses, dict):
        # VAE with multiple loss components
        plt.subplot(1, 2, 1)
        plt.plot(losses['total'], label='Total Loss')
        plt.plot(losses['recon'], label='Reconstruction')
        plt.plot(losses['kl'], label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(losses['total'])
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title(f'{model_name} - Total Loss')
        plt.grid(True, alpha=0.3)
    else:
        # Simple loss list
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - Training Loss')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cluster_quality(results):
    """
    Plot elbow curve and silhouette scores for cluster selection.

    Args:
        results: Dictionary from find_optimal_clusters()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    if len(results['inertias']) > 0:
        axes[0].plot(results['k_values'], results['inertias'], 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True, alpha=0.3)

    # Silhouette scores
    axes[1].plot(results['k_values'], results['silhouette_scores'], 'ro-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].axhline(y=0.5, color='g', linestyle='--', label='Good separation (>0.5)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Recommend best k
    best_k_idx = np.argmax(results['silhouette_scores'])
    best_k = results['k_values'][best_k_idx]
    best_score = results['silhouette_scores'][best_k_idx]

    print(f"\nRecommended k: {best_k} (silhouette score: {best_score:.3f})")

def visualize_clusters_by_metadata(embeddings, metadata_df, color_by='cluster',
                                method='pacmap', title=None, figsize=(15, 10),
                                save_path=None):

    print(f"Creating 2D {method.upper()} projection...")

    # Reduce to 2D
    if method == 'pacmap':
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Check if color_by column exists
    if color_by not in metadata_df.columns:
        print(f"Warning: Column '{color_by}' not found in metadata_df")
        print(f"Available columns: {metadata_df.columns.tolist()}")
        # Plot without coloring
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.7, s=50, c='gray')
    else:
        color_values = metadata_df[color_by].copy()

        # Handle NaN values
        has_nan = color_values.isna().any()
        if has_nan:
            n_nan = color_values.isna().sum()
            print(f"Warning: {n_nan} NaN values in '{color_by}' column")

        # Determine if categorical or continuous
        is_categorical = (
            color_values.dtype == 'object' or
            color_by in ['cluster', 'view', 'subject'] or
            len(color_values.unique()) <= 20  # Treat as categorical if ≤20 unique values
        )

        if is_categorical:
            # Categorical - use discrete colors
            # Get unique values (excluding NaN)
            unique_vals = color_values.dropna().unique()

            # Sort for consistent ordering
            try:
                unique_vals = sorted(unique_vals)
            except TypeError:
                # Can't sort mixed types, keep as is
                unique_vals = list(unique_vals)

            n_colors = len(unique_vals)
            print(f"Plotting {n_colors} categories: {unique_vals}")

            # Choose color palette based on number of categories
            if n_colors <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_colors]
            elif n_colors <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_colors]
            else:
                # Use continuous colormap for many categories
                colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))

            # Plot each category
            for i, val in enumerate(unique_vals):
                mask = (color_values == val).values
                if mask.sum() > 0:
                    ax.scatter(
                        coords_2d[mask, 0],
                        coords_2d[mask, 1],
                        c=[colors[i]],
                        label=f'{val} (n={mask.sum()})',
                        alpha=0.7,
                        s=50,
                        edgecolors='none'
                    )

            # Plot NaN values separately if they exist
            if has_nan:
                nan_mask = color_values.isna().values
                ax.scatter(
                    coords_2d[nan_mask, 0],
                    coords_2d[nan_mask, 1],
                    c='lightgray',
                    label=f'Unknown (n={nan_mask.sum()})',
                    alpha=0.5,
                    s=50,
                    marker='x',
                    edgecolors='none'
                )

            # Add legend
            if n_colors <= 30:
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    fontsize=8,
                    frameon=True,
                    fancybox=True,
                    shadow=True
                )
            else:
                print(f"Too many categories ({n_colors}) to show legend")

        else:
            # Continuous - use colorbar
            # Remove NaN values for plotting
            valid_mask = ~color_values.isna()

            scatter = ax.scatter(
                coords_2d[valid_mask, 0],
                coords_2d[valid_mask, 1],
                c=color_values[valid_mask],
                cmap='viridis',
                alpha=0.7,
                s=50,
                edgecolors='none'
            )

            # Plot NaN as gray
            if has_nan:
                nan_mask = color_values.isna().values
                ax.scatter(
                    coords_2d[nan_mask, 0],
                    coords_2d[nan_mask, 1],
                    c='lightgray',
                    alpha=0.5,
                    s=50,
                    marker='x',
                    label=f'Unknown (n={nan_mask.sum()})',
                    edgecolors='none'
                )
                ax.legend(loc='upper right')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_by, fontsize=12)

    # Labels and title
    if title is None:
        title = f'{method.upper()} Projection Colored by {color_by}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()

    return coords_2d


# ## Embedding Helper

# In[ ]:


def batch_convert_lif_to_npy(lif_dir, npy_dir):
    lif_dir = Path(lif_dir)
    npy_dir = Path(npy_dir)

    converted = 0
    failed = 0

    for lif_path in lif_dir.rglob('*.lif'):
        try:
            lif = LifFile(str(lif_path))
            img = lif.get_image(0)

            channels = []
            for ch in range(img.channels):
                z_slices = [np.array(img.get_frame(z=z, t=0, c=ch)) for z in range(img.dims.z)]
                channels.append(np.stack(z_slices, axis=0))
            data = np.stack(channels, axis=0)  # (C, Z, H, W)

            # Preserve folder structure
            relative = lif_path.relative_to(lif_dir)
            out_path = npy_dir / relative.with_suffix('.npy')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, data)

            if hasattr(lif, '_f'):
                lif._f.close()

            converted += 1
            print(f"OK ({converted}): {relative} → shape {data.shape}")

        except Exception as e:
            failed += 1
            print(f"FAIL: {lif_path.name} - {e}")

    print(f"\nDone. Converted: {converted}, Failed: {failed}")


def get_embedding_multi_channel(img_tensor, embedder, resize_to=(224, 224)):
    """
    Generate embedding from multi-channel image.

    Args:
        img_tensor: Tensor of shape [C, H, W]
        embedder: Multi-channel PyTorch model
        resize_to: Target size for model input

    Returns:
        Embedding vector of shape [embedding_dim]
    """
    transform = transforms.Compose([
        transforms.Resize(resize_to),
    ])

    # Add batch dimension: [C, H, W] → [1, C, H, W]
    img_batch = img_tensor.unsqueeze(0)
    img_resized = transform(img_batch)

    # Generate embedding
    with torch.no_grad():
        embedding = embedder(img_resized)

    # Remove batch dimension: [1, embedding_dim] → [embedding_dim]
    return embedding.squeeze(0)


def embedding_to_numpy(embedding_tensor):
    """
    Convert PyTorch tensor embedding to numpy array.

    Args:
        embedding_tensor: PyTorch tensor (can be on GPU or CPU)

    Returns:
        numpy array (always on CPU)
    """
    # Move to CPU if on GPU, then convert to numpy
    return embedding_tensor.mps().detach().numpy()


# In[ ]:


def process_image_to_embedding(embedder, key, data_dir, ch_order, export_embeddings=False, convert_to_tensor=False):
    all_arrays = []
    for dir_name in os.listdir(data_dir):
            # check if dir_name is in key ID column
            if dir_name in key["ID"].values:
                # groups = key[key["ID"] == dir_name]["Group"].values[0]
                continue
            else:
                print(f"{dir_name} not found in key")

            for file_name in os.listdir(os.path.join(data_dir, dir_name)):
                if file_name.endswith(".npy"):
                    # freq = file_name.split(".")[6]+"."+file_name.split(".")[7]
                    # view = file_name.split(".")[8]

                    file_path = os.path.join(data_dir, dir_name, file_name)
                    tensor = convert_max_proj_tensor(file_path, ch_order)
                    normalized_tensor = normalize_per_channel(tensor)
                    embedding = get_embedding_multi_channel(normalized_tensor, embedder)
                    embedding_numpy = embedding_to_numpy(embedding)
                    all_arrays.append(embedding_numpy)

                    if export_embeddings:
                        np.save(f'embedding_v1.npy', embedding_numpy) #figure out naming convention

    all_embeddings = np.stack(all_arrays)

    if convert_to_tensor:
        torch_batch = torch.from_numpy(all_embeddings).float()
        print(f"Batch of embeddings{all_embeddings.shape} (batch_size=2, embedding_dim=512)\n[NOTE: Returned as torch tensor]")
        return torch_batch

    print(f"Batch of embeddings: {all_embeddings.shape} (batch_size=2, embedding_dim=512)")

    return all_embeddings
                # how to stack and export tensors effectively?

                # out_zarr = os.path.join("./zarrs", "liberman", group.replace(" ", "_"), f"{dir_name}_{freq}_{view}.zarr")
                # if not os.path.exists(os.path.join(out_zarr, 'ctbp2')):
                #     count += 1
                #     print(count, f"Saving {file_name} to {out_zarr}")
                #     out = zarr.open(out_zarr, 'a')

# np.load('.npy') to load later                    


# In[ ]:


def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for all images in dataloader.

    Args:
        model: Trained ShallowCNN encoder
        dataloader: DataLoader (can be with or without augmentation)
        device: 'mps', 'cuda', or 'cpu'

    Returns:
        np.array: Stacked embeddings [num_images, embedding_dim]
    """
    model = model.to(device)
    model.eval()

    embeddings = []
    skipped_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Skip empty batches (all images failed to load)
            if batch.numel() == 0:
                skipped_batches += 1
                continue

            # Handle paired data (if augment=True)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    if skipped_batches > 0:
        print(f"⚠️  Skipped {skipped_batches} empty batches during embedding extraction")

    if len(embeddings) == 0:
        print("⚠️  WARNING: No valid embeddings extracted! All images failed to load.")
        return np.array([])

    return np.vstack(embeddings)


# ## Embedding-Metadata Integration

# In[ ]:


# ============================================================================
# Image Metadata Extraction and Clustering Integration
# ============================================================================
# Connect clustering results with experimental metadata to interpret findings

import re
import os

def parse_image_metadata(file_path):
    """
    Extract metadata from .lif file path.

    File naming convention:
    [SubjectID]L.CtBP2.GluR2.NF.Myo7.IHC.[Frequency].[View].lif
    Example: WPZ116L.CtBP2.GluR2.NF.Myo7.IHC.45.2.v1.lif

    Args:
        file_path: Full path to .lif file

    Returns:
        dict with keys: 'subject', 'frequency', 'view', 'file_path'
    """
    filename = os.path.basename(file_path)

    # Extract subject (e.g., WPZ116)
    subject_match = re.match(r'(WPZ\d+)', filename)
    subject = subject_match.group(1) if subject_match else None

    # Extract frequency (e.g., 45.2 or 8.0)
    freq_match = re.search(r'IHC\.(\d+\.\d+|\d+)\.v', filename)
    frequency = float(freq_match.group(1)) if freq_match else None

    # Extract view (v1 or v2)
    view_match = re.search(r'\.(v[12])\.npy', filename)
    view = view_match.group(1) if view_match else None

    return {
        'Subject': subject,
        'Frequency': frequency,
        'View': view,
        'file_path': file_path,
        'filename': filename
    }


def create_metadata_dataframe(image_paths, experimental_metadata):
    """
    Create DataFrame with metadata for all images.

    Args:
        image_paths: List of .lif file paths
        experimental_metadata: Can be:
            - Path to WPZ Mouse groups.xlsx (string)
            - Pre-loaded DataFrame with subject metadata
            - None (will only parse image paths)

    Returns:
        pandas DataFrame with columns: subject, frequency, view, file_path,
        and optionally: Group, Strain, and other metadata columns
    """
    # Parse all image paths
    image_metadata = []
    for path in image_paths:
        meta = parse_image_metadata(path)  # Subject, Frequency, vx, file_path, filename
        image_metadata.append(meta)

    image_metadata_df = pd.DataFrame(image_metadata)

    exp_meta = experimental_metadata.copy()

    merge_cols = ['Subject', 'Frequency', 'View' 'Strain', 'Group', 'Hours Elapsed Post-Exposure', 'dB Noise Exposure', 'Amplitude', 'Synapses to IHC']

    merged_df = image_metadata_df.merge(
        exp_meta[merge_cols],
        on=['Subject', 'Frequency', 'View'],
        how='inner'
    )

    return merged_df


def merge_clusters_with_metadata(embeddings, cluster_labels, image_paths,
                                  experimental_metadata):
    """
    Merge clustering results with experimental metadata.
    Joins on [subject, frequency, view].

    Args:
        embeddings: numpy array [n_samples, embedding_dim]
        cluster_labels: numpy array [n_samples]
        image_paths: List of .lif file paths
        experimental_metadata: DataFrame with metadata

    Returns:
        DataFrame with metadata + clusters
    """
    print(f"\nMerging clusters with metadata...")
    print(f"  Images: {len(image_paths)}, Clusters: {len(cluster_labels)}")

    # Parse image paths to get subject, frequency, view
    image_metadata_list = []
    for idx, path in enumerate(image_paths):
        meta = parse_image_metadata(path)
        meta['Cluster'] = cluster_labels[idx]
        for i in range(min(embeddings.shape[1], 10)):
            meta[f'emb_dim_{i}'] = embeddings[idx, i]
        image_metadata_list.append(meta)

    image_metadata_df = pd.DataFrame(image_metadata_list)
    # print(f"  Base: {image_metadata_df.shape}, columns: {image_metadata_df.columns.tolist()}")

    exp_meta = experimental_metadata.copy()

    print(f"  Experimental: {exp_meta.shape}")

    # Merge on [subject, frequency, view]
    merged = exp_meta.merge(image_metadata_df, on=['Subject', 'Frequency', 'View'], how='right') # prioritize the image databank

    # print(f"  Merged: {merged.shape} ({len(image_metadata_df)} images -> {len(merged)} rows)")
    print(f"  Cluster distribution:\n{merged['cluster'].value_counts()}")

    return merged





def analyze_cluster_composition(metadata_df, cluster_col='cluster',
                                groupby_cols=['Group', 'Strain']):
    """
    Analyze what metadata characteristics are in each cluster.

    Args:
        metadata_df: DataFrame from merge_clusters_with_metadata()
        cluster_col: Column name with cluster labels
        groupby_cols: List of metadata columns to analyze

    Returns:
        Dictionary of DataFrames showing composition of each cluster
    """
    results = {}

    for col in groupby_cols:
        if col in metadata_df.columns:
            # Count distribution
            composition = metadata_df.groupby([cluster_col, col]).size().unstack(fill_value=0)

            # Calculate percentages
            composition_pct = composition.div(composition.sum(axis=1), axis=0) * 100

            results[col] = {
                'counts': composition,
                'percentages': composition_pct
            }

            print(f"\n{'='*60}")
            print(f"Cluster Composition by {col}")
            print('='*60)
            print("\nCounts:")
            print(composition)
            print("\nPercentages:")
            print(composition_pct.round(1))
        else:
            print(f"\nWarning: Column '{col}' not found in metadata_df")
            print(f"Available columns: {metadata_df.columns.tolist()}")

    return results


# In[ ]:


def analyze_cluster_composition_v2(metadata_df, cluster_col='cluster',
                                groupby_cols=['Group', 'Strain']):
    """
    Analyze what metadata characteristics are in each cluster.

    Args:
        metadata_df: DataFrame from merge_clusters_with_metadata()
        cluster_col: Column name with cluster labels
        groupby_cols: List of metadata columns to analyze

    Returns:
        Dictionary of DataFrames showing composition of each cluster
    """
    results = {}

    for col in groupby_cols:
        if col in metadata_df.columns:
            # Count distribution
            composition = metadata_df.groupby([cluster_col, col]).size().unstack(fill_value=0)

            # Calculate percentages
            composition_pct = composition.div(composition.sum(axis=1), axis=0) * 100

            results[col] = {
                'counts': composition,
                'percentages': composition_pct
            }

            print(f"\n{'='*60}")
            print(f"Cluster Composition by {col}")
            print('='*60)
            print("\nCounts:")
            print(composition)
            print("\nPercentages:")
            print(composition_pct.round(1))

    return results


# ## Non-Image Data Pipeline (see pkl file)

# ### Basic Metadata

# In[ ]:


# # Leah Mac/Unix directories
# # abr_path = '/Users/leahashebir/Downloads/Manor_Practicum/liberman_data/Ephys Data Charles Liberman'
# # image_dir = "/Users/leahashebir/Downloads/Manor_Practicum/liberman_data/Confocal Data Charles Liberman/"

# # Manor Windows directories
# abr_path = Path('O:\BPHO Dropbox\Manor Lab\ManorLab\File requests\liberman_lab_data\Paired_Ephys_SynapseConfocal\Ephys Data Charles Liberman')


# In[ ]:


# filter1 = 128
# filter2 = 32
# dropout1 = 0.5
# dropout2 = 0.3
# dropout_fc = 0.1

# # Model initialization
# peak_finding_model = CNN(filter1, filter2, dropout1, dropout2, dropout_fc)
# model_loader = torch.load('./models/waveI_cnn.pth')
# peak_finding_model.load_state_dict(model_loader)
# peak_finding_model.eval()


# In[ ]:


# time_scale = 18
# amp_per_freq = {'Subject': [], 'Latencies' : [], 'Freq(kHz) (x1)': [], 'Level(dB) (x2)': [], 'Amplitude (x3)':[]}

# for subject in os.listdir(abr_path):
#     if subject.startswith('WPZ'):
#         print(repr(subject))
#         for fq in os.listdir(os.path.join(abr_path,subject)):
#             print('frequency:',repr(fq))
#             if fq.startswith('ABR') and fq.endswith('.tsv'):
#                 abr_file = os.path.join(abr_path,subject,fq)
#                 data_df = read_custom_tsv(abr_file)
#                 # print(data_df)
#                 freqs = data_df['Freq(kHz)'].unique().tolist()
#                 levels = data_df['Level(dB)'].unique().tolist()
#                 for freq in freqs:
#                     for lvl in levels:
#                         highest_peaks, y_values, amp = peaks_troughs_amp_final(df=data_df, freq=freq, db=lvl, time_scale=time_scale)
#                         latencies = latency_all_peaks(highest_peaks, y_values, time_scale)
#                         if len(latencies) < 5:
#                             print(subject, freq , latencies)
#                             continue
#                         amp_per_freq['Subject'].append(subject)
#                         amp_per_freq['Freq(kHz) (x1)'].append(freq)
#                         amp_per_freq['Level(dB) (x2)'].append(lvl)
#                         amp_per_freq['Amplitude (x3)'].append(amp)
#                         amp_per_freq['Latencies'].append(latencies)
#             else:
#                 pass


# amp_df_full = pd.DataFrame(data=amp_per_freq)

# raw_synapse_counts = pd.read_excel('/Users/leahashebir/Downloads/Manor_Practicum/liberman_data/WPZ Ribbon and Synapse Counts.xlsx')
# # raw_synapse_counts = raw_synapse_counts.mask(lambda x: x.isnull()).dropna() # old approach 
# raw_synapse_counts = raw_synapse_counts.mask(lambda x: x.isna(),0)
# raw_synapse_counts['Synapses to IHC (y1)'] = raw_synapse_counts.iloc[:,6]
# raw_synapse_counts['vx (x4)'] = raw_synapse_counts['vx']
# raw_synapse_counts.drop(columns=['vx'], inplace=True)
# raw_synapse_counts.rename(columns={'Freq':'Freq(kHz) (x1)'}, inplace=True)
# raw_synapse_counts.rename(columns={'Case':'Subject'}, inplace=True)


# In[ ]:


# # Version 1 - values per vx

# paired = amp_df_full.join(raw_synapse_counts.set_index(['Subject', 'Freq(kHz) (x1)']), on=['Subject', 'Freq(kHz) (x1)'])
# # slice = paired[paired['Subject']=='WPZ174'][['Subject', 'Freq(kHz) (x1)', 'Level(dB) (x2)', 'Amplitude (x3)', 'vx (x4)','Synapses to IHC (y1)', 'IHCs (y2)']]
# final = paired[['Subject', 'Latencies', 'Freq(kHz) (x1)', 'Level(dB) (x2)', 'Amplitude (x3)', 'vx (x4)','Synapses to IHC (y1)', 'IHCs']]
# final_clean = final.dropna()

# # adding in the strain feature
# strains = pd.read_excel('/Users/leahashebir/Downloads/Manor_Practicum/liberman_data/WPZ Mouse groups.xlsx')
# final_clean_strained = final_clean.join(strains.set_index('ID#'), on='Subject')
# final_clean_strained['Strain'] = final_clean_strained['Strain'].str.strip()
# final_clean_strained = final_clean_strained.rename(columns={'Strain': 'Strain (x5)'})
# final_clean_strained = final_clean_strained.dropna()
# final_clean_strained = final_clean_strained[['Subject', 'Latencies', 'Freq(kHz) (x1)', 'Level(dB) (x2)', 'Amplitude (x3)', 'vx (x4)', 'Strain (x5)', 'Synapses to IHC (y1)', 'Group']]

# final_clean_strained_grouped = final_clean_strained.copy()
# final_clean_strained_grouped['Group - dB'] = final_clean_strained_grouped['Group'].apply(lambda x: x.split(' ')[0] if x.split(' ')[0].endswith('dB') else 'Control')
# final_clean_strained_grouped['Group - Time Elapsed'] = final_clean_strained_grouped['Group'].apply(lambda x: x.split(' ')[1] if x.split(' ')[1].endswith(('h', 'wks', 'w')) else x.split(' ')[0])
# final_clean_strained_grouped.head()

# final_clean_strained_grouped_pos = final_clean_strained_grouped.copy()
# final_clean_strained_grouped_pos['Amplitude (x3)'] = final_clean_strained_grouped['Amplitude (x3)'].apply(lambda x: 0 if x < 0 else x)

# final_clean_strained_grouped_pos_cleangroup = final_clean_strained_grouped_pos.copy()
# final_clean_strained_grouped_pos_cleangroup['Group'] = final_clean_strained_grouped_pos_cleangroup['Group'].apply(lambda x: x.strip())

# final_clean_strained_grouped_pos_cleangroup.head()
# final_clean_strained_grouped_pos_cleangroup_vs = final_clean_strained_grouped_pos_cleangroup.copy()
# final_clean_strained_grouped_pos_cleangroup_vs['Group - dB (x6)'] = final_clean_strained_grouped_pos_cleangroup_vs['Group - dB']
# # final_clean_strained_grouped_pos_cleangroup_vs['Group - Time Elapsed (x7)'] = final_clean_strained_grouped_pos_cleangroup_vs['Group - Time Elapsed']
# final_clean_strained_grouped_pos_cleangroup_vs = final_clean_strained_grouped_pos_cleangroup_vs[['Subject','Latencies', 'Freq(kHz) (x1)', 'Level(dB) (x2)', 'Amplitude (x3)',
#        'vx (x4)', 'Strain (x5)','Group - dB (x6)', 'Group - Time Elapsed', 'Group','Synapses to IHC (y1)']]

# def split_on_number(input_string):
#     return re.findall(r"[A-Za-z]+|\d+", input_string)

# hrs_week = 24*7

# final_clean_strained_grouped_pos_cleangroup_vs_timed = final_clean_strained_grouped_pos_cleangroup_vs.copy()
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'].apply(lambda x: '0dB' if x == 'Control' else x)
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'].apply(split_on_number)
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'].apply(lambda x: x[0])
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - dB (x6)'].apply(lambda x: int(x.strip()))
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Split'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed'].apply(split_on_number)
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Magn.'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Split'].apply(lambda x: x[0])
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Magn.'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Magn.'].apply(lambda x: int(x.strip()))
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Unit'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Split'].apply(lambda x: x[1])
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Unit'] = final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Time Elapsed - Unit'].apply(lambda x: "wks" if x == 'w' else x)
# final_clean_strained_grouped_pos_cleangroup_vs_timed['Group - Hours Elapsed (x7)'] = final_clean_strained_grouped_pos_cleangroup_vs_timed.apply(lambda row: row['Group - Time Elapsed - Magn.']* hrs_week if row['Group - Time Elapsed - Unit'] == 'wks' else row['Group - Time Elapsed - Magn.'], axis = 1)

# final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded = final_clean_strained_grouped_pos_cleangroup_vs_timed.copy()
# final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Strain encoded (x5)'] = final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Strain (x5)'].apply(lambda x: 0 if x=='C57B6' else 1)
# final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded.head()


# In[ ]:


# np.unique(final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Strain encoded (x5)'] ), np.unique(final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Strain (x5)'] )


# In[ ]:


# time_scale = 18
# subject_ABRs = {}

# for subject in os.listdir(abr_path):
#     if subject in raw_synapse_counts['Subject'].values: # excluding subjects not in synapse count file
#         for fq in os.listdir(os.path.join(abr_path,subject)):
#             if fq.startswith('ABR') and fq.endswith('.tsv'):
#                 match = re.search(r'-L-([\d.]+)\.tsv$', fq)
#                 if match:
#                     freq = float(match.group(1))
#                     if freq in raw_synapse_counts[raw_synapse_counts['Subject'] == subject]['Freq(kHz) (x1)'].values:
#                         # if freq == 6.0 or freq == 7.0:
#                         #     print(subject, freq)
#                         # freqs.add(freq)
#                         path = os.path.join(abr_path,subject,fq)
#                         data_df = read_custom_tsv(path)
#                         if data_df['Freq(kHz)'].iloc[0] == freq:
#                             subject_ABRs[(subject, freq)] = data_df
#                         else:
#                             print(f"Skipping subject {subject}, frequency {freq} due to mismatch.")


# In[ ]:


# final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded.head()


# ### Metadata for Clustering

# In[ ]:


# final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Freq(kHz) (x1)'].unique()


# In[ ]:


# # Building df specific to clustering analysis: removing level?

# subject_metadata_for_cluster_analysis = final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded.copy()
# subject_metadata_for_cluster_analysis = subject_metadata_for_cluster_analysis.loc[:, ['Subject', 'Freq(kHz) (x1)', 'Level(dB) (x2)', 'vx (x4)', 'Strain (x5)', 'Group - dB (x6)','Group - Hours Elapsed (x7)', 'Synapses to IHC (y1)']].reset_index()
# subject_metadata_for_cluster_analysis = subject_metadata_for_cluster_analysis.rename(columns={'Freq(kHz) (x1)': 'Frequency', 'vx (x4)': 'View', 'Strain (x5)': 'Strain', 'Group - dB (x6)': 'dB Noise Exposure', 'Group - Hours Elapsed (x7)': 'Hours Elapsed Post-Exposure', 'Synapses to IHC (y1)': 'Synapses to IHC'})
# subject_metadata_for_cluster_analysis.loc[(subject_metadata_for_cluster_analysis['Subject'] == 'WPZ174') & (subject_metadata_for_cluster_analysis['Frequency'] == 8.0) & (subject_metadata_for_cluster_analysis['View'] == 'v1'), :]

# subject_metadata_for_cluster_analysis_collapsed = subject_metadata_for_cluster_analysis.drop_duplicates(subset=['Subject', 'Frequency', 'View'], keep='first')
# subject_metadata_for_cluster_analysis_collapsed.loc[subject_metadata_for_cluster_analysis_collapsed['Subject'] == 'WPZ133', :]
# # subject_metadata_for_cluster_analysis_collapsed.head()


# ### Waveform Data

# In[ ]:


# penalty = L2Regularization(linear_operator=LinearDifferentialOperator(2))
# shared_grid = np.linspace(0, time_scale, 244)
# # basis = BSplineBasis(n_basis=7, domain_range=(0,18)) # old
# # smoother = BasisSmoother(basis=basis, return_basis=True) # , regularization=penalty, smoothing_parameter=0.1, 
# time_scale = 18
# subjects = []
# frequencies = []
# levels = []
# strains = []
# amps = []
# raw_waves = []
# newbasis_waves = []
# Xs = []
# ys = []
# fails = []
# bases = []
# all_latencies = []

# for (sub, freq), df in subject_ABRs.items():
#     for lvl in np.unique(df['Level(dB)']):
#         latencies_series = final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded[(final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Subject'] == sub)\
#             & (final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Freq(kHz) (x1)'] == freq)\
#                 & (final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Level(dB) (x2)'] == lvl)]\
#             ['Latencies']

#         if len(latencies_series) == 0:
#             print(f'N/A latencies: ({sub}, {freq}, {lvl}) : {latencies_series}')
#             continue

#         latencies = latencies_series.values[0]
#         latencies = [float(x) for x in latencies]
#         all_latencies.append(latencies)
#         # print(latencies)

#     for lvl in np.unique(df['Level(dB)']):
#         strain_series = final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded[(final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Subject'] == sub)\
#             & (final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Freq(kHz) (x1)'] == freq)\
#                 & (final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Level(dB) (x2)'] == lvl)]\
#             ['Strain (x5)']

#         if len(strain_series) == 0:
#             print(f'Strain error, none recorded: ({sub}, {freq}, {lvl}) : {strain_series}')
#             continue

#         amp_series = final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded[(final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Subject'] == sub)\
#                 & (final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Freq(kHz) (x1)'] == freq)]['Amplitude (x3)']
#         if len(amp_series) == 0:
#             print(f'Amplitude error, none recorded: ({sub}, {freq}, {lvl}) : {amp_series}')

#         amp = float(amp_series.values[0])

#         lvl = float(lvl)
#         strain = strain_series.values[0]
#         wave = full_interpolation(df, freq, lvl, time_scale)
#         wave = np.asarray(wave, dtype=float)
#         wave = wave.reshape(1, -1)

#         grid = time_scale * np.arange(0, 244) / 244
#         wave_input = skfda.FDataGrid(data_matrix=wave,grid_points=shared_grid)

#         basis = BSplineBasis(domain_range=(latencies[0], latencies[-1]),knots = latencies, order = 3)
#         bases.append(basis)
#         smoother = BasisSmoother(basis=basis, return_basis=True, regularization=penalty, smoothing_parameter=1.0000e+02)

#         wave_newbasis = smoother.fit_transform(wave_input)[0] # smoother will allow regularization for further tuning down the line...

#         X = wave_newbasis.coefficients

#         y_series = final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded[(final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Subject'] == sub)\
#             & (final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Freq(kHz) (x1)'] == freq)\
#             & (final_clean_strained_grouped_pos_cleangroup_vs_timed_strain_encoded['Level(dB) (x2)'] == lvl)]\
#             ['Synapse to IHC Ratio per Freq (y2)']

#         # print(y_series)

#         if len(y_series) == 0 or pd.isna(y_series.iloc[0]):
#             print(f'N/A y: ({sub}, {freq}, {lvl})')
#             continue

#         y = float(y_series.iloc[0])

#         # print((sub, freq, lvl, y))
#         subjects.append(sub)
#         frequencies.append(freq)
#         levels.append(lvl)
#         strains.append(strain)
#         amps.append(amp)
#         raw_waves.append(wave_input)
#         newbasis_waves.append(wave_newbasis)
#         Xs.append(X.flatten()) # used for model fitting, same as for OLS!
#         ys.append(y)


# final_waves_df_new_basis_best = pd.DataFrame(data = {'Subject' : subjects, 'Freq(kHz)' : frequencies, 'Level(dB)' : levels, 'Strain' : strains, 'Amplitude' : amps, 'Latencies' : all_latencies, 'Raw Waves' : raw_waves, 'Transformed Waves (X)' : Xs, 'Synapse to IHC Ratio per Freq (y2)' : ys})
# final_waves_df_new_basis_best_clean = final_waves_df_new_basis_best.dropna().reset_index(drop=True)
# final_waves_df_new_basis_best_clean.head()


# # 1. Shallow CNN Embedding Pipeline

# ## Converting to Npy (do this once!)

# In[ ]:


image_dir_source = Path(r'\O:\BPHO Dropbox\Manor Lab\ManorLab\File requests\liberman_lab_data\Paired_Ephys_SynapseConfocal\Confocal Data Charles Liberman')
image_dir_npy = Path(r'D:\Leah\Liberman Data\npy_conversions_confocal_data')
# batch_convert_lif_to_npy(image_dir_source, image_dir_npy)  # ONLY DO THIS ONCE


# ## CUDA Set-up

# In[ ]:


image_paths = collect_all_npy_paths(image_dir_npy)
device = 'cuda' if torch.cuda.is_available() else 'mps'
# print(f"Using device: {device}")  # This should be printing cuda!!!!
global_num_workers = 0


# ## Testing file types

# In[ ]:


# import time
# import numpy as np
# import zarr
# import tifffile
# from readlif.reader import LifFile
# from pathlib import Path

# # --- Step 1: Convert 5 .lif files to other formats ---
# image_paths_testing = collect_all_lif_paths(image_dir_source)
# test_files = image_paths_testing[:5]
# save_dir = Path(r'D:\Leah\format_test')
# save_dir.mkdir(parents=True, exist_ok=True)

# npy_paths = []
# zarr_paths = []
# tiff_paths = []

# for i, lif_path in enumerate(test_files):
#     print(f"Converting file {i+1}/5: {os.path.basename(lif_path)}")

#     lif = LifFile(lif_path)
#     img = lif.get_image(0)

#     channels = []
#     for ch in range(img.channels):
#         z_slices = [np.array(img.get_frame(z=z, t=0, c=ch)) for z in range(img.dims.z)]
#         channels.append(np.stack(z_slices, axis=0))
#     data = np.stack(channels, axis=0)  # (C, Z, H, W)

#     # Save each format
#     npy_path = save_dir / f'test_{i}.npy'
#     np.save(npy_path, data)
#     npy_paths.append(npy_path)

#     zarr_path = save_dir / f'test_{i}.zarr'
#     zarr.save(str(zarr_path), data)
#     zarr_paths.append(zarr_path)

#     tiff_path = save_dir / f'test_{i}.tiff'
#     tifffile.imwrite(str(tiff_path), data)
#     tiff_paths.append(tiff_path)

# print(f"\nData shape: {data.shape}, dtype: {data.dtype}")
# print("All conversions done. Starting benchmark...\n")

# # --- Step 2: Load each of the 5 files per format, measure time ---
# def bench_lif(path):
#     lif = LifFile(path)
#     img = lif.get_image(0)
#     channels = []
#     for ch in range(img.channels):
#         z_slices = [np.array(img.get_frame(z=z, t=0, c=ch)) for z in range(img.dims.z)]
#         channels.append(np.stack(z_slices, axis=0))
#     return np.stack(channels, axis=0)

# def bench_npy(path):
#     return np.load(path)

# def bench_zarr(path):
#     return zarr.load(str(path))

# def bench_tiff(path):
#     return tifffile.imread(str(path))

# tests = {
#     'LIF':   (bench_lif, test_files),
#     'NumPy': (bench_npy, npy_paths),
#     'Zarr':  (bench_zarr, zarr_paths),
#     'TIFF':  (bench_tiff, tiff_paths),
# }

# print(f"{'Format':<10} {'Mean (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
# print("-" * 46)

# for name, (func, paths) in tests.items():
#     times = []
#     for path in paths:
#         start = time.perf_counter()
#         arr = func(path)
#         elapsed = time.perf_counter() - start
#         times.append(elapsed)
#     print(f"{name:<10} {np.mean(times):<12.4f} {np.min(times):<12.4f} {np.max(times):<12.4f}")


# In[ ]:


# # can i convert to npy from online copies:

# from readlif.reader import LifFile
# from pathlib import Path
# import numpy as np
# import os

# test_path = r'O:\BPHO Dropbox\Manor Lab\ManorLab\File requests\liberman_lab_data\Paired_Ephys_SynapseConfocal\Confocal Data Charles Liberman\WPZ101\WPZ101L.CtBP2.GluR2.NF.Myo7.IHC.32.0.v1.lif'
# npy_dir = Path(r'D:\Leah\npy_data')
# print(f"Testing: {test_path}")

# try:
#     lif = LifFile(test_path)
#     img = lif.get_image(0)

#     channels = []
#     for ch in range(img.channels):
#         z_slices = [np.array(img.get_frame(z=z, t=0, c=ch)) for z in range(img.dims.z)]
#         channels.append(np.stack(z_slices, axis=0))
#     data = np.stack(channels, axis=0)

#     # Save to local drive
#     out_path = npy_dir / 'test.npy'
#     npy_dir.mkdir(parents=True, exist_ok=True)
#     np.save(out_path, data)

#     # Verify it loads back
#     loaded = np.load(out_path)
#     print(f"Success! Shape: {loaded.shape}, Size: {os.path.getsize(out_path) / 1e6:.1f} MB")

# except OSError as e:
#     print(f"FAILED - file is still online-only: {e}")
#     print("You need the local copies to convert. Don't delete them yet.")


# In[ ]:


# # Checking for no data loss in npy conversion

# lif = LifFile(test_path)
# img = lif.get_image(0)

# loaded_npy = np.load(out_path)

# # Check shape
# print(f"Channels: {img.channels}, Z: {img.dims.z}")
# print(f"NPY shape: {loaded_npy.shape}")  # should be (C, Z, H, W)

# # Verify each channel matches
# for ch in range(img.channels):
#     for z in range(img.dims.z):
#         original = np.array(img.get_frame(z=z, t=0, c=ch))
#         saved = loaded_npy[ch, z]
#         assert np.array_equal(original, saved), f"Mismatch at ch={ch}, z={z}"

# print("All channels and z-slices match exactly.")


# ## Fine tuning

# ### Finetuning Model

# In[ ]:


from xml.parsers.expat import model


class ShallowCNN_Tuning:
    """
    Factory class for building ShallowCNN model + dataloader with hyperparameter tuning.

    Usage:
        builder = ShallowCNN_Tuning(
            image_paths=lif_paths,
            embedding_dim=256,
            learning_rate=0.001,
            temperature=0.3,
            batch_size=4
        )
        model, dataloader = builder.build()
        losses, trained_model = builder.train(model, dataloader, epochs=10)
    """
    def __init__(self, image_paths, input_channels=4, embedding_dim=256, 
                 learning_rate=0.001, temperature=0.3, batch_size=4, 
                 num_workers=4, optimizer='Adam', target_size=224):
        # Store hyperparameters
        self.image_paths = image_paths
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size  
        self.optimizer = optimizer

    def build(self, dataset, device='cuda'):
        """
        Build and return dataloader.

        Model and dataset built separately now.

        Returns:
            eval_dataloader: DataLoader for embedding extraction
        """

        # Create dataloader for embedding extraction
        eval_dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return eval_dataloader

    def train(self, model, dataset, epochs, device='cuda'):
        """
        Train the model with contrastive learning.

        Args:
            model: ShallowCNN instance
            dataset: SynapseImageDataset instance
            epochs: Number of training epochs
            device: Device to train on

        Returns:
            losses: List of average losses per epoch
            model: Trained encoder (without projection head)
        """
        model = model.to(device)
        scaler = torch.amp.GradScaler(device)

        # Create projection head
        projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 128)
        ).to(device)

        # Optimizer for both model and projection head
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(projection_head.parameters()),
                lr=self.learning_rate
            )
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                list(model.parameters()) + list(projection_head.parameters()),
                lr=self.learning_rate
            )

        model.train()
        projection_head.train()
        losses = []

        valid_pairs = dataset.get_all_valid_pairs()
        print(f"Valid pairs: {len(valid_pairs)}")

        for epoch in range(epochs):
            total_loss = 0
            valid_batches = 0

            # Shuffle pairs each epoch
            random.shuffle(valid_pairs)

            for i in range(0, len(valid_pairs), self.batch_size):
                batch_pairs = valid_pairs[i:i+self.batch_size]
                if len(batch_pairs) < 2:
                    continue

                original_batch = torch.stack([dataset.cache[p[0]] for p in batch_pairs]).to(device)
                paired_batch = torch.stack([dataset.cache[p[1]] for p in batch_pairs]).to(device)

                with torch.amp.autocast('cuda'):
                    emb1 = model(original_batch)
                    emb2 = model(paired_batch)
                    proj1 = projection_head(emb1)
                    proj2 = projection_head(emb2)
                    loss = self._contrastive_loss(proj1, proj2)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                valid_batches += 1


        # for epoch in range(epochs):
        #     total_loss = 0
        #     num_batches = len(dataset) // self.batch_size
        #     valid_batches = 0

        #     for batch_idx in range(num_batches):
        #         # Sample random indices
        #         indices = torch.randperm(len(dataset))[:self.batch_size]

        #         original_image_list = []
        #         paired_image_list = []

        #         # For each image, get its pair
        #         for idx in indices:
        #             img, pair_img = dataset.get_pair(idx.item())

        #             if img is not None and pair_img is not None:
        #                 original_image_list.append(img)
        #                 paired_image_list.append(pair_img)

        #         # Skip batch if too few pairs found
        #         if len(paired_image_list) < 2:
        #             continue

        #         original_image_batch = torch.stack(original_image_list).to(device)
        #         paired_image_batch = torch.stack(paired_image_list).to(device)

        #         # Forward pass through encoder
        #         emb1 = model(original_image_batch)
        #         emb2 = model(paired_image_batch)

        #         # Forward pass through projection head
        #         proj1 = projection_head(emb1)
        #         proj2 = projection_head(emb2)

        #         # Compute contrastive loss
        #         loss = self._contrastive_loss(proj1, proj2)

        #         # Backward pass
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()


                # Mixed precision approach to speed up training - UPDATE failed :( made training unstable, loss diverged to NaN

                # with torch.amp.autocast(device):
                #     # Forward pass through encoder
                #     emb1 = model(original_image_batch)
                #     emb2 = model(paired_image_batch)

                #     # Forward pass through projection head
                #     proj1 = projection_head(emb1)
                #     proj2 = projection_head(emb2)

                #     # Compute contrastive loss
                #     loss = self._contrastive_loss(proj1, proj2)

                # Backward pass
                # optimizer.zero_grad()
                # scaler.scale(loss).backward()

                # scaler.step(optimizer)
                # scaler.update()
                # scaler.unscale_(optimizer)


                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_loss += loss.item()
                valid_batches += 1

            avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Valid batches: {valid_batches}")

        return losses, model  # Return encoder only, not projection head

    def _contrastive_loss(self, emb1, emb2):
        """
        Contrastive loss (NT-Xent / SimCLR style).

        Args:
            emb1, emb2: Embeddings from original and paired images [batch_size, embedding_dim]

        Returns:
            Scalar loss value
        """
        batch_size = emb1.shape[0]

        # L2 normalize
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        # Concatenate
        embeddings = torch.cat([emb1, emb2], dim=0)  # [2*batch, dim]

        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.T) / self.temperature

        # Create labels
        labels = torch.cat([
            torch.arange(batch_size) + batch_size,  # emb1 → emb2
            torch.arange(batch_size)                # emb2 → emb1
        ], dim=0).to(emb1.device)

        # Mask diagonal
        mask = torch.eye(2*batch_size, dtype=torch.bool, device=emb1.device)
        similarity = similarity.masked_fill(mask, -65000)

        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        return loss


# Keep ShallowCNN as a clean model architecture (no training logic inside)
class ShallowCNN(nn.Module):
    """
    Lightweight 3-layer CNN encoder.

    Architecture:
        Input: 4-channel 224x224 images
        Conv1: 4 → 32 channels (3x3)
        Conv2: 32 → 64 channels (3x3)
        Conv3: 64 → 128 channels (3x3)
        Global pooling → embedding_dim output
    """
    def __init__(self, input_channels=4, embedding_dim=256):
        super(ShallowCNN, self).__init__()

        # Conv blocks with AvgPool
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(2, 2)  # 224 → 112

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(2, 2)  # 112 → 56

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(2, 2)  # 56 → 28

        self.gap = nn.AdaptiveAvgPool2d(1)  # 28x28x128 → 1x1x128
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        # x: [batch, 4, 224, 224]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.gap(x)  # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]
        x = self.fc(x)  # [batch, embedding_dim]

        return x


# ### Finetuning Loop

# In[ ]:


# todays_ft = f'3_21' # update this per ft round
# Path(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\models').mkdir(parents=True, exist_ok=False)
# Path(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\embeddings').mkdir(parents=True, exist_ok=False)

# np.random.seed(42)
# torch.manual_seed(42)

# learning_rates = [0.05, 0.1, 0.15]
# temperatures = [0.1, 0.15, 0.2]
# embedding_dims = [256, 512]
# # batches = [4, 8, 16]
# batch = 128  # still may need tuning...

# all_results = {}
# best_db = 0
# best_config = None
# model_index = 0

# for learning_rate in learning_rates:
#     for temperature in temperatures:
#         for emb_dim in embedding_dims:
#             model_index += 1
#             print("\n" + "="*60)
#             print(f"Model {model_index}: Learning rate: {learning_rate}, Temperature: {temperature}")
#             print("="*60)

#             builder = ShallowCNN_Tuning(
#                 image_paths=image_paths,
#                 embedding_dim=emb_dim,
#                 learning_rate=learning_rate,
#                 temperature=temperature,
#                 batch_size=batch,
#                 num_workers=global_num_workers
#             )

#             model, paired_dataset, eval_dataloader = builder.build()
#             losses, trained_model = builder.train(model, paired_dataset, epochs=10, device=device)
#             embeddings = extract_embeddings(trained_model, eval_dataloader, device=device)
#             torch.save(trained_model, rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\models\model_{model_index}.pth')
#             np.save(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\embeddings\embeddings_{model_index}.npy', embeddings)

#             results = find_optimal_clusters(embeddings, max_k=10, method='kmeans')
#             plot_cluster_quality(results)
#             best_k_idx = np.argmax(results['silhouette_scores'])
#             k_optimal = results['k_values'][best_k_idx]
#             sil_score = results['silhouette_scores'][best_k_idx]
#             cluster_labels, cluster_model = cluster_embeddings(embeddings, method='kmeans', n_clusters=k_optimal)
#             db_score = davies_bouldin_score(embeddings, cluster_labels)
#             if db_score > best_db:
#                 best_db = db_score
#                 best_config = (model_index, learning_rate, temperature)

#             all_results[f'Model {model_index}'] = {
#                 'learning_rate': learning_rate,
#                 'temperature': temperature,
#                 'final_loss': losses[-1],
#                 'best_k': k_optimal,
#                 'silhouette_score': sil_score,
#                 'db_score': db_score,
#                 'losses': losses,
#                 'model': trained_model,
#                 'emb_dimensions': emb_dim,
#                 'batch_size': batch
#             }
#             plot_training_curves(losses, model_name=f'Shallow CNN - Model {model_index}')
#             print(f"Model data saved successfully. Model parameters: {sum(p.numel() for p in trained_model.parameters()):,}. Silhouette Score for best k={k_optimal}: {sil_score:.4f}. Davies-Bouldin Score: {db_score:.4f}.")

# all_results.to_pickle(r'D:\Leah\unsupervised_clustering\unsupervised_confocal_analysis\finetuning\{todays_ft}\final_results.pkl')
# print(f'Best Shallow CNN Config Based on Davies-Bouldin Score: Model {best_config[0]} - LR={best_config[1]}, Temp={best_config[2]}')


# In[ ]:


# all_results


# ### Best Model Results

# In[ ]:


# experimental_metadata = pd.readpickle('abr_metadata_per_freq.pkl')
# best_model_embeddings = np.load(f'/Users/leahashebir/Downloads/Manor_Practicum/modeling/unsupervised_confocal_analysis/finetuning/{todays_ft}/embeddings/embeddings_{best_config[0]}.npy')
# best_model_cluster_labels, _ = cluster_embeddings(best_model_embeddings, method='kmeans', n_clusters=2)

# best_model_results = merge_clusters_with_metadata(
#     embeddings=best_model_embeddings,
#     cluster_labels=best_model_cluster_labels,
#     image_paths=image_paths,
#     experimental_metadata=experimental_metadata
# )

# best_model_results = best_model_results[['cluster','Subject', 'Frequency', 'View', 'Strain',  'Hours Elapsed Post-Exposure', 'dB Noise Exposure', 'Synapses to IHC', 'file_path']]

# # coords = visualize_clusters_by_metadata(best_model_embeddings, best_model_results, color_by='Strain')
# # coords = visualize_clusters_by_metadata(best_model_embeddings, best_model_results, color_by='Frequency')
# # coords = visualize_clusters_by_metadata(best_model_embeddings, best_model_results, color_by='View')


# ### Resulting Image Projections

# In[ ]:


# print('============================================ Cluster 0 Examples ============================================')
# import random
# for i in random.sample(range(len(best_model_results)), 3):
#     image_mask = best_model_results.loc[best_model_results['cluster'] == 0].iloc[i]
#     path = image_mask.loc['file_path']

#     print(os.path.basename(image_mask['file_path']))
#     print(image_mask[['Strain', 'Frequency', 'View']])
#     reader = AICSImage(path)
#     img = reader.get_image_data("ZCYX")

#     max_proj = np.max(img, axis=0)

#     composite = np.zeros((*max_proj.shape[1:], 3)) 
#     colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)]

#     for c in range(max_proj.shape[0]):
#         channel = max_proj[c].astype(np.float64)
#         channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
#         for rgb in range(3):
#             composite[:, :, rgb] += channel * colors[c][rgb]

#     composite = np.clip(composite, 0, 1)

#     plt.figure(figsize=(10, 10))
#     plt.imshow(composite)
#     plt.title("Composite Max Projection")
#     plt.axis('off')
#     plt.show()


# print('\n\n\n\n\n============================================ Cluster 1 Examples ============================================')  
# for i in range(3):
#     image_mask = best_model_results.loc[best_model_results['cluster'] == 1].iloc[i]
#     path = image_mask.loc['file_path']

#     print(os.path.basename(image_mask['file_path']))
#     print(image_mask[['Strain', 'Frequency', 'View']])
#     reader = AICSImage(path)
#     img = reader.get_image_data("ZCYX")

#     max_proj = np.max(img, axis=0)

#     composite = np.zeros((*max_proj.shape[1:], 3))  # (Y, X, 3)
#     colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)]  # R, G, B, Magenta...

#     for c in range(max_proj.shape[0]):
#         channel = max_proj[c].astype(np.float64)
#         channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
#         for rgb in range(3):
#             composite[:, :, rgb] += channel * colors[c][rgb]

#     composite = np.clip(composite, 0, 1)

#     plt.figure(figsize=(10, 10))
#     plt.imshow(composite)
#     plt.title("Composite Max Projection")
#     plt.axis('off')
#     plt.show()


# # Main

# In[ ]:


# if __name__ == '__main__':
#     image_paths = collect_all_npy_paths(image_dir_npy)
#     device = 'cuda' if torch.cuda.is_available() else 'mps'
#     print(f"Using device: {device}")  # This should be printing cuda!!!!
#     # global_num_workers = 16

#     todays_ft = f'3_21' # update this per ft round
#     Path(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\models').mkdir(parents=True, exist_ok=False)
#     Path(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\embeddings').mkdir(parents=True, exist_ok=False)

#     np.random.seed(42)
#     torch.manual_seed(42)

#     learning_rates = [0.0001, 0.0005, 0.001]
#     temperatures = [0.1, 0.15, 0.2]
#     embedding_dims = [256, 512]
#     # batches = [4, 8, 16]
#     batch = 128  # still may need tuning...

#     all_results = {}
#     model_index = 0

#     paired_dataset = SynapseImageDataset(
#         image_paths=image_paths,
#         target_size=224)

#     print(f"Images with pairs: {len(paired_dataset.pairs)}")
#     for learning_rate in learning_rates:
#         for temperature in temperatures:
#             for emb_dim in embedding_dims:
#                 model_index += 1
#                 print("\n" + "="*60)
#                 print(f"Model {model_index}: Learning rate: {learning_rate}, Temperature: {temperature}")
#                 print("="*60)

#                 builder = ShallowCNN_Tuning(
#                     image_paths=image_paths,
#                     embedding_dim=emb_dim,
#                     learning_rate=learning_rate,
#                     temperature=temperature,
#                     batch_size=batch,
#                     num_workers=global_num_workers
#                 )

#                 eval_dataloader = builder.build()

#                 model = ShallowCNN(
#                     input_channels=4,
#                     embedding_dim=emb_dim
#                 )

#                 # Saving model
#                 cl_losses, trained_model = builder.train(model, paired_dataset, epochs=1000, device=device)
#                 torch.save(trained_model, rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\models\model_{model_index}.pth')

#                 # Saving embeddings
#                 embeddings = extract_embeddings(trained_model, eval_dataloader, device=device)
#                 np.save(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\embeddings\embeddings_{model_index}.npy', embeddings)

#                 # Assessing clusters via kmeans
#                 results = find_optimal_clusters(embeddings, max_k=10, method='kmeans')
#                 plot_cluster_quality(results)
#                 best_k_idx = np.argmax(results['silhouette_scores'])
#                 k_optimal = results['k_values'][best_k_idx]
#                 k_sil_score = results['silhouette_scores'][best_k_idx]
#                 k_cluster_labels = cluster_embeddings(embeddings, method='kmeans', n_clusters=k_optimal)
#                 k_db_score = davies_bouldin_score(embeddings, k_cluster_labels)

#                 # Assessing clusters via HDBSCAN
#                 hdb_labels = cluster_embeddings(embeddings, method='hdbscan', min_cluster_size=10)
#                 n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
#                 n_noise = (hdb_labels == -1).sum()
#                 print(f"HDBSCAN found {n_clusters_hdb} clusters, {n_noise} noise points")

#                 # Only score if HDBSCAN found >1 cluster
#                 if n_clusters_hdb > 1:
#                     # Exclude noise points for scoring
#                     valid = hdb_labels != -1
#                     hdb_sil = silhouette_score(embeddings[valid], hdb_labels[valid])
#                     hdb_db = davies_bouldin_score(embeddings[valid], hdb_labels[valid])
#                     print(f"HDBSCAN - Silhouette: {hdb_sil:.4f}, DB: {hdb_db:.4f}")
#                 else:
#                     hdb_sil = -1
#                     hdb_db = float('inf')

#                 # Saving all model metadata and clustering results
#                 all_results[f'Model {model_index}'] = {
#                     'learning_rate': learning_rate,
#                     'temperature': temperature,
#                     'emb_dimensions': emb_dim,
#                     'batch_size': batch,
#                     'final_loss': cl_losses[-1],
#                     'best_k': k_optimal,
#                     'k_silhouette_score': k_sil_score,
#                     'k_db_score': k_db_score,
#                     'hdb_silhouette_score': hdb_sil,
#                     'hdb_db_score': hdb_db,
#                     'losses': cl_losses,

#                 }
#                 plot_training_curves(cl_losses, model_name=f'Shallow CNN - Model {model_index}')
#                 print(f"Model data saved successfully. Model parameters: {sum(p.numel() for p in trained_model.parameters()):,}. Silhouette Score for best k={k_optimal}: {sil_score:.4f}. Davies-Bouldin Score: {db_score:.4f}.")

#     with open(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\final_results.pkl', 'wb') as f:
#         pickle.dump(all_results, f)
#         print(f'All results saved successfully.')


# In[ ]:


if __name__ == '__main__':
    image_paths = collect_all_npy_paths(image_dir_npy)
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")  # This should be printing cuda!!!!

    todays_ft = f'3_22' # update this per ft round
    Path(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\models').mkdir(parents=True, exist_ok=False)
    Path(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\embeddings').mkdir(parents=True, exist_ok=False)

    np.random.seed(42)
    torch.manual_seed(42)

    learning_rates = [0.0001, 0.0005, 0.001]
    temperatures = [0.1, 0.15, 0.2]
    embedding_dims = [256, 512]
    # batches = [4, 8, 16]
    batch = 128  # still may need tuning...

    model_results = {}
    model_index = 0

    paired_dataset = SynapseImageDataset(
        image_paths=image_paths,
        target_size=224)

    print(f"Starting SIMCLR-Shallow CNN training for images with pairs: {len(paired_dataset.pairs)}")
    for learning_rate in learning_rates:
        for temperature in temperatures:
            for emb_dim in embedding_dims:
                model_index += 1
                print("\n" + "="*60)
                print(f"Model {model_index}: Learning rate: {learning_rate}, Temperature: {temperature}")
                print("="*60)

                builder = ShallowCNN_Tuning(
                    image_paths=image_paths,
                    embedding_dim=emb_dim,
                    learning_rate=learning_rate,
                    temperature=temperature,
                    batch_size=batch,
                    num_workers=global_num_workers
                )

                eval_dataloader = builder.build(dataset=paired_dataset, device=device)

                model = ShallowCNN(
                    input_channels=4,
                    embedding_dim=emb_dim
                )

                # Saving model
                cl_losses, trained_model = builder.train(model, paired_dataset, epochs=1000, device=device)
                torch.save(trained_model, rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\models\model_{model_index}.pth')

                # Saving embeddings
                embeddings = extract_embeddings(trained_model, eval_dataloader, device=device)
                np.save(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\embeddings\embeddings_{model_index}.npy', embeddings)

                # Saving all model metadata and clustering results
                model_results[f'Model {model_index}'] = {
                    'learning_rate': learning_rate,
                    'temperature': temperature,
                    'emb_dimensions': emb_dim,
                    'batch_size': batch,
                    'final_loss': cl_losses[-1],
                    'losses': cl_losses   
                }
                print(f"Model data saved successfully. Model parameters: {sum(p.numel() for p in trained_model.parameters()):,}")

    with open(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\model_training_results.pkl', 'wb') as f:
        pickle.dump(model_results, f)
        print(f'All model results saved successfully.')

    clustering_results = {}

    print("="*60, "Starting Clustering Analysis...", "="*60)
    for model in model_results.keys():

        # Grabbing saved embeddings for analysis
        model_idx = int(model.split()[1])
        embeddings = np.load(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\embeddings\embeddings_{model_idx}.npy')

        # Assessing clusters via kmeans
        results = find_optimal_clusters(embeddings, max_k=10, method='kmeans')
        best_k_idx = np.argmax(results['silhouette_scores'])
        k_optimal = results['k_values'][best_k_idx]
        k_sil_score = results['silhouette_scores'][best_k_idx]
        k_cluster_labels = cluster_embeddings(embeddings, method='kmeans', n_clusters=k_optimal)
        k_db_score = davies_bouldin_score(embeddings, k_cluster_labels)

        # Assessing clusters via HDBSCAN
        hdb_labels = cluster_embeddings(embeddings, method='hdbscan', min_cluster_size=10)
        n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = (hdb_labels == -1).sum()
        print(f"HDBSCAN found {n_clusters_hdb} clusters, {n_noise} noise points")

        # Only score if HDBSCAN found >1 cluster
        if n_clusters_hdb > 1:
            # Exclude noise points for scoring
            valid = hdb_labels != -1
            hdb_sil = silhouette_score(embeddings[valid], hdb_labels[valid])
            hdb_db = davies_bouldin_score(embeddings[valid], hdb_labels[valid])
            print(f"HDBSCAN - Silhouette: {hdb_sil:.4f}, DB: {hdb_db:.4f}")
        else:
            hdb_sil = -1
            hdb_db = float('inf')

        # Saving clustering results
        clustering_results[model] = {
            'kmeans_k': k_optimal,
            'kmeans_silhouette': k_sil_score,
            'kmeans_db': k_db_score,
            'hdbscan_clusters': n_clusters_hdb,
            'hdbscan_noise': int(n_noise),
            'hdbscan_silhouette': hdb_sil,
            'hdbscan_db': hdb_db
            }

    # Saving clustering results
    with open(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\clustering_results.pkl', 'wb') as f:
        pickle.dump(clustering_results, f)
        print(f'All clustering results saved successfully.')

    # Merging model training and clustering results
    model_clustering_results = {k: {**model_results[k], **clustering_results[k]} for k in model_results}
    with open(rf'D:\Leah\unsupervised_clustering\finetuning\{todays_ft}\model_clustering_results.pkl', 'wb') as f:
        pickle.dump(model_clustering_results, f)
        print(f'Merged model and clustering results saved successfully.')

