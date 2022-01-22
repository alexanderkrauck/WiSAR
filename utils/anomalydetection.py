"""
Author: Cai Yitao
Date: 2022/1/22
"""

import torch.nn.functional as F
import os
import json
from torchvision import datasets
import cv2
from utils.data import MultiViewTemporalSample, make_impossible_mask
import utils.data as data
import utils.sub_architectures as sub_architectures
import utils.architectures as architectures
from utils.basic_function import draw_labels
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch.utils.data import DataLoader
from utils.basic_function import integrate_images
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import softmax
import scipy
import ot



class IntegrateReconstructedImages(object):
    """modified from MultiviewTemporalSample"""
    def __init__(self,path):
        self.path = path
    
        homography_dict = json.load(
            open(os.path.join(path, "homographies.json"))
        )
        _photo_order = ["B05", "B04", "B03", "B02", "B01", "G01", "G02", "G03", "G04", "G05"]

        self.homographies = []

        self.path = path
        for timestep in range(0, 7):

            timestep_homographies = []
            for perspective in _photo_order:
                name = str(timestep) + "-" + perspective

                homography = homography_dict[name]

                timestep_homographies.append(homography)

            timestep_homographies = np.array(timestep_homographies)

            self.homographies.append(timestep_homographies)

        self.homographies = np.array(self.homographies)
    def integrate(self, photos, timestep):
        return integrate_images(photos[timestep], self.homographies[timestep])



class WassersteinAnomalyScore(object):
    def __init__(self,path,sample:data.MultiViewTemporalSample,
    integrateimages:IntegrateReconstructedImages,
    square_distance = True):
        self.path = path
        self.integrateimages = integrateimages

        self.sample = sample
        self.square_distance = square_distance


    def get_score(self,reconstructed_photos):
    
        scores = []
        for t in range(7):
            int_img = self.sample.integrate(t)
            int_recon_img = self.integrateimages.integrate(reconstructed_photos, t)
            if self.square_distance:
                distance = np.square(abs(int_img - int_recon_img))
                score = cv2.GaussianBlur(np.amax(distance,axis=-1),(9,9),6)
                scores.append(score)
            else:
                distance = w_score(int_img,int_recon_img)
                score = cv2.GaussianBlur(distance,(9,9),6)
                scores.append(score)
        scores = np.array(scores)
        
        # scores = np.mean(scores,axis=0)
        return np.amax(scores,axis=0)

def w_score(image,reconstruct_image):
    m,n,_ = image.shape
    wass_score = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            wass_score[i,j] = wasserstein_distance(image[i,j,:],reconstruct_image[i,j,:])

    return wass_score


def reconstruct_image(image, autoencoder):
   

    photos = image.photos
    reconstructed_photos = np.empty_like(photos)

    for timestep_idx, timestep in enumerate(photos):
        for view_idx, view in enumerate(timestep):
            grid_images = []
            reconstructed_view = np.empty_like(view)
            for row in range(16):
                for col in range(16):
                    grid_images.append(
                        view[
                            row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64
                        ]
                    )

            grid_images = np.array(grid_images)
            grid_images = (
                np.transpose(grid_images, (0, 3, 1, 2)).astype(np.float32) / 255
            )

            grid_images = torch.tensor(grid_images, device="cpu")
            reconstructed_grid_images = (
                autoencoder(grid_images)
                .detach().cpu()
                .numpy()
            )
            reconstructed_grid_images = np.transpose(
                (reconstructed_grid_images * 255).astype(np.int16), (0, 2, 3, 1)
            )

            i = 0
            for row in range(16):
                for col in range(16):
                    reconstructed_view[
                        row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64
                    ] = reconstructed_grid_images[i]
                    i += 1
            reconstructed_photos[timestep_idx, view_idx] = reconstructed_view
    return reconstructed_photos


class MSEAnomalyDetection(object):
    def __init__(self,score,threshold = 0.5):
        self.threshold = threshold
        self.score = score
    def get_anomaly_bb(self,sample):
        mask = make_impossible_mask(self.sample)
        self.score[mask] = 0
        self.score[self.score < 0.2 *255] = 0
        
        thresh = ((self.score >= self.threshold) * 255).astype(np.uint8)
        contours,_= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cntr in contours:
            box = np.array(cv2.boundingRect(cntr))
            boxes.append(box.tolist())
        return boxes

def collect_labels(dataset,path,autoencoder,w_bbs_f):
    # dirs = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    # dirs.sort()
    labels = {}
             
    for sample in dataset:
        sample_name = os.path.split(sample.sample_path)[-1]
      
        sub_path = os.path.join(path, sample_name)
        reconstructed_photos = reconstruct_image(sample,autoencoder)
        integrateimages = IntegrateReconstructedImages(sub_path)
        anoscore = WassersteinAnomalyScore(sub_path,sample,integrateimages)
        score = anoscore.get_score(reconstructed_photos)
        box = MSEAnomalyDetection(score)
        boxes = box.get_anomaly_bb(sample)
        labels[sample_name] = boxes
    
    if w_bbs_f is not None:
        with open(w_bbs_f, "w") as f:
            json.dump(labels, f)
        
    return labels
