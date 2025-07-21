import os
import random
from tqdm import tqdm

import numpy as np
from sklearn.cluster import OPTICS, AgglomerativeClustering
from functools import reduce

import torch
import torch.nn.functional as F
import cv2 as cv

from src.utils import mlt_process, CHECK_INPUT_CUDA
from HETMM_cuda import PTS

from feb import get_feb

def gen_by_ALL(model, temploader, tpath, backbone, half=False, save=True):
    Out_dict = {}
    # print (f'Generating the original template')
    with torch.no_grad():
        for batch in temploader:
            x = batch[0].cuda().half() if half else batch[0].cuda()
            for k, v in model.make_template(x).items():
                Out_dict[k] = [v.cuda()] if k not in Out_dict else Out_dict[k] + [v.cuda()]
        Out_dict = {k : torch.cat(v, dim=0) for k, v in Out_dict.items()}

    # Train FEB Model for Spesific Category
    category = os.path.basename(tpath)
    if category in ['bottle', 'capsule', 'metal_nut', 'toothbrush']:
        train_features = Out_dict.get('layer1')
        print(f"Features Shape: {train_features.shape}")
        feb = get_feb(train_features).to(x.device).eval()
        
        save_path = os.path.join(tpath, f'{backbone}_ALL_FEB.pkl')
        torch.save(feb.state_dict(), save_path)
        print(f"[INFO] FEB state_dict saved to: {save_path}")
    
    if save:
        torch.save(Out_dict, os.path.join(tpath, f'{backbone}_ALL.pkl'))
    return Out_dict

def gen_by_PC_OPTICS(tdict, tsize, tpath, backbone, num_workers, save=True, **kwargs):
    clu_path = os.path.join(tpath, f'{backbone}_PC_OPTICSx{tsize}.pkl')
    if os.path.exists(clu_path):
        # print ('Found pretrained OPTICS clusters!')
        clu_dict = torch.load(clu_path, weights_only=True, map_location='cpu')

    else:
        clu_dict = get_pixel_level_optics_clusters(tdict, tsize, num_workers) 

        if save:
            torch.save(clu_dict, clu_path)

    return clu_dict

def gen_by_PC_Agglomerative(tdict, tsize, tpath, backbone, num_workers, save=True, **kwargs):
    clu_path = os.path.join(tpath, f'{backbone}_PC_Agglomerativex{tsize}.pkl')
    if os.path.exists(clu_path):
        clu_dict = torch.load(clu_path, weights_only=True, map_location='cpu')
    else:
        clu_dict = get_pixel_level_agglomerative_clusters(tdict, tsize, num_workers)
        if save:
            torch.save(clu_dict, clu_path)
    return clu_dict

def gen_by_PTS(tdict, tsize, tpath, backbone, num_workers, save=True, **kwargs):

    # clu_dict = gen_by_PC_OPTICS(tdict, tsize, tpath, backbone, num_workers, save=True, **kwargs)
    clu_dict = gen_by_PC_Agglomerative(tdict, tsize, tpath, backbone, num_workers, save=True, **kwargs)

    Out_dict = {}
    # print ('Generating PTS...')
    for key, Ts in tqdm(tdict.items()):
        Out_dict[key] = PTS(Ts, clu_dict[key].to(Ts.device).long(), int(tsize))

    if save:
        torch.save(Out_dict, os.path.join(tpath, f'{backbone}_PTSx{tsize}.pkl'))

    return Out_dict

def get_pixel_level_optics_clusters(tdict, clu_size, num_workers):

    def get_optics_clusters_unit(pixel, min_samples, metric):
        optics = OPTICS(min_samples=min_samples, metric=metric)
        return optics.fit_predict(pixel.copy(order='C'))[..., np.newaxis]

    clu_dict = {}

    # print ('Generating OPTICS clusters (very slow)...')
    for key, Ts in tqdm(tdict.items()):
        N, C, H, W = Ts.shape
        pixels = F.unfold(Ts, 1).cpu().numpy()
        const_params = [max(2, N // clu_size), lambda x1, x2 : 1 - x1.dot(x2.T)]
        params = [[pixels[..., i]] for i in range(H * W)]
        clusters = mlt_process(get_optics_clusters_unit, params, const_params, num_workers)
        clu_dict[key] = torch.from_numpy(np.concatenate(clusters, -1)).reshape(N, H, W).float()

    return clu_dict

def get_pixel_level_agglomerative_clusters(tdict, clu_size, num_workers):


    def get_agglomerative_clusters_unit(pixel, n_clusters, metric):
        """
        Clustering patch features using Agglomerative Clustering.
    
        Args:
            pixel: ndarray [N, C] – patch-level features
            n_clusters: int – number of clusters (tsize)
            metric: str – affinity metric ('euclidean', 'cosine', etc.)
    
        Returns:
            labels: ndarray [N, 1]
        """
        pixel = pixel.astype(np.float64)
            
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage='average')
    
        labels = clusterer.fit_predict(pixel)
        return labels[..., np.newaxis]


    clu_dict = {}

    for key, Ts in tqdm(tdict.items()):
        N, C, H, W = Ts.shape
        pixels = F.unfold(Ts, 1).cpu().numpy()  # shape: [N, C, H*W]
        const_params = [clu_size, 'cosine']
        params = [[pixels[..., i]] for i in range(H * W)]  # each = [N, C]
        clusters = mlt_process(get_agglomerative_clusters_unit, params, const_params, num_workers)
        clu_dict[key] = torch.from_numpy(np.concatenate(clusters, -1)).reshape(N, H, W).float()
    
    return clu_dict
