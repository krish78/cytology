# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:53:57 2023

@author: Lucid
"""
import cv2
import torch
import os
import numpy as np

import openslide

from openslide.deepzoom import DeepZoomGenerator

import random


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


class load_custom(Dataset):
    def __init__(self, path, img_size , tile_size, overlap, batch_size = 16 , single_cls = False, rect = False ,
                                  stride = 32, pad = 0.0):
        print('in here')
        self.img_size = img_size
        self.rect = rect
        self.stride = stride
        # self.tile_source = tiles       
        # self.level = level
        self.path = path
        # create all col and row indexes in series in an array
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        
        self.img_files = []
        
        
        
        slide = openslide.open_slide(path)
        tile_source = DeepZoomGenerator(slide, tile_size = tile_size, overlap = overlap, limit_bounds=False) 
        self.level = tile_source.level_count -1
        cols, rows = tile_source.level_tiles[self.level]        
        print(path, tile_size, batch_size, self.level)
        print(cols, rows)        
        for row in range(rows):            
            for col in range(cols):
                 self.img_files.append((col,row))                                         
        n = int(cols*rows)  # number of images/tiles        
        bi = int(n/batch_size) #np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        print('tiles :', n,' batchsize :', batch_size, 'nos batches :', bi)
        #nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)
        print('load over')
         
    def __len__(self):
             return len(self.img_files)

         
    def __getitem__(self, index):
            index = self.indices[index]  # linear, shuffled, or image_weights

                # Load image
            img, coord = load_tile(self, index)

            img, ratio, pad = letterbox(img,auto=False)#, scaleup=self.augment)
            
            # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.transpose(img , (2, 0, 1)).astype(np.float16)
            img = np.expand_dims(img, 0)
            img = np.ascontiguousarray(img)
            return img , self.img_files[index], ratio , coord
        
    def collate_fn(batch):
            img, indices, ratios , coords = zip(*batch)  # transposed            
            return img, ratios , coords
    

def load_tile(self, index):
    
    tile_id = self.img_files[index]
    slide = openslide.open_slide(self.path)
    tile_source = DeepZoomGenerator(slide, tile_size = self.tile_size, overlap= self.overlap, limit_bounds=False) 
    img = np.array(tile_source.get_tile(tile_source.level_count-1,tile_id))    
    #img = np.ascontiguousarray(img)
    coord = tile_source.get_tile_coordinates(tile_source.level_count-1, tile_id)                #assert img is not None, 'Image Not Found ' + path


    return img, coord   # img, hw_original, hw_resized








