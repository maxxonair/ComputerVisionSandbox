#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 09:25:33 2022

@author: x
"""

class FeatureMatches:
    
    keypoint_img1_xy = []
    keypoint_img2_xy = []
    
    px1_xy = []
    px2_xy = []
    
    def set_px1(self, keypoint):
        self.px1_xy = ( int(keypoint[0]), int(keypoint[1]))
        
    def set_px2(self, keypoint):
        self.px2_xy = ( int(keypoint[0]), int(keypoint[1]))
    
        
    