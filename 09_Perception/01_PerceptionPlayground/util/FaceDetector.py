'''
This file contains a set of functions to support facial recogition tasks using 
facenet_pytorch

'''
import cv2 as cv
import os
from time import perf_counter
import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Dataclass imports 
from dataclasses import dataclass
from typing import Tuple
from typing import List

import util.image_functions as img
import util.constants as cnst

@dataclass
class MonoFace:
  """ Class for keeping track of key metrics from a single camera face detection """
  id: int
  box_top_left_px: Tuple[int, int]
  box_bottom_right_px: Tuple[int, int]
  center_px: Tuple[int, int]
  confidence: float
  img: List[int]
  
  def __init__(self, id, box_top_left_px, box_bottom_right_px, center_px, confidence, img):
    self.id = id
    self.box_top_left_px = box_top_left_px
    self.box_bottom_right_px = box_bottom_right_px
    self.center_px = center_px
    self.confidence = confidence
    self.img = img

@dataclass
class StereoFace:
  """ Class for keeping track of key metrics from a stereo face detection """
  id: int
  center_coords_left_px: Tuple[int, int]
  center_coords_right_px: Tuple[int, int]
  radius_px: int
  confidence: float
  
  def __init__(self, id, center_coords_left_px, center_coords_right_px, radius_px, confidence):
    self.id = id
    self.center_coords_left_px = center_coords_left_px
    self.center_coords_right_px = center_coords_right_px
    self.radius_px = radius_px
    self.confidence = confidence
   
class FaceKey:
  """ Class to maintain key features of a face from a known person """
  name_id: str
  img_template: List[int]
  embeddings = 0
  
  def __init__(self, name_id, img_in):
    self.name_id = name_id
    self.img_template = img_in
    faces_template, _ = self.mtcnn.detect(img_in)
    
    if faces_template is None:
      print('[ERROR] No faces found in template')
    else:
      aligned_template = self.mtcnn(faces_template)
      self.embeddings = self.resnet(aligned_template).detach()
      
  def is_match(self, img_in):
    '''
    Function to find a face match in a given image
    
    returns: flag (bool) True if match is found. False otherwise
    '''
    # Detect faces and extract embeddings
    faces_img, _ = self.mtcnn.detect(img_in)
    isMatchFound = False

    if faces_img is None:
      self.log.pLogErr('No faces found in image')
    else:
      aligned_img = self.mtcnn(faces_img)
      embeddings_img = self.resnet(aligned_img).detach()
      
      # Compare the Euclidean distance between embeddings to matching threshold
      if (((embeddings_img - self.embeddings).norm().item()) 
          < self.THR_EMBED_DISTANCE_FACE_MATCHING):  
        self.log.pLogMsg("Match found.")
        isMatchFound = True
      else:
        self.log.pLogWrn("No match found.")
    return isMatchFound

class FaceDetector():
  #  -----  [ Class Data ]
  FACE_MARKER_THICKNESS = 1
  FACE_MARKER_COLOR     = (0,240,2)
  
  # Set confidence threshold to discard face recognition results
  FACE_RECOGN_CONF_THR  = 0.8
  
  # Default prefix when saving stereo frames to file
  IMG_PREFIX = 'stereoimg_'
  
  # Define HUD settings
  HUD_FONT_SCALE = 0.4
  HUD_TEXT_START_OFFSET_Y = 40
  HUD_TEXT_START_OFFSET_X = 20
  HUD_TEXT_STEP_Y = 20
  HUD_FONT = cv.FONT_HERSHEY_SIMPLEX
  
  # Threshold for euclidian distance between embeddings to assess a face match
  THR_EMBED_DISTANCE_FACE_MATCHING = 0.95
  
  #  -----  [ Class Init ]
  def __init__(self, log):
    self.log = log
    # Setup facial recognition pipeline
    # Run on GPU if available. On CPU otherwise
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    
    # Initialize MTCNN for face detection
    self.mtcnn = MTCNN(keep_all=True, device=device)
    
    # Load pre-trained Inception ResNet model
    self.resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    
    
  def detect(self, img_in):
    """
    Perform face recogition and store all detected faces in a list of MonoFaces

    Args:
        img_in (np.array(np.int8)): Input image

    Returns:
       numDetections int32 : Number of succesful detetions
       faceList list : List of all successful detections as MonoFace instances
       displayImg_out (np.array(np.int8)) : Output images with marked detections
       
    """
    # Create copies to draw on
    displayImg_out = img_in.copy()
      
    # Detect faces in the input image
    boxes, conficence = self.mtcnn.detect(img_in)
    
    # Initialize list to store successful detections
    faceList = []
    
    # If any matches have been found
    if boxes is not None:
      # Loop over all detected faces
      for faceIndx in range(len(conficence)):
        if conficence[faceIndx] > self.FACE_RECOGN_CONF_THR:
          boxes = np.asarray(boxes, np.int32)
          
          # Compute the coordinates of the center of the detected face
          faceCenterCoords_px = (int((boxes[0,0] + boxes[0,2]) / 2),
                                int((boxes[0,1] + boxes[0,3]) / 2))
          
          tempSf = MonoFace(id=faceIndx,
                            box_top_left_px=(boxes[faceIndx,0],boxes[faceIndx,1]),
                            box_bottom_right_px=(boxes[faceIndx,2],boxes[faceIndx,3]),
                            center_px=faceCenterCoords_px,
                            confidence=conficence,
                            img=img_in[boxes[faceIndx,1]:boxes[faceIndx,3],
                                       boxes[faceIndx,0]:boxes[faceIndx,2]])
          
          # Append data class to list of detections
          faceList.append(tempSf)         
              
          # Draw face outline as a box 
          displayImg_out = cv.rectangle(displayImg_out,
                                        tempSf.box_top_left_px,
                                        tempSf.box_bottom_right_px,
                                        self.FACE_MARKER_COLOR,
                                        self.FACE_MARKER_THICKNESS)
          # Draw the center of the detected face
          displayImg_out = cv.circle(displayImg_out,faceCenterCoords_px,
                                      10, self.FACE_MARKER_COLOR, 1)
          displayImg_out = cv.circle(displayImg_out,faceCenterCoords_px,
                                      1, self.FACE_MARKER_COLOR, 1)
          
    numDetections = len(faceList)
    return numDetections, faceList, displayImg_out
        
  def match(self, img_in, template_in):
    '''
    Function to find a face match in a given images
    
    returns: flag (bool) True if match is found. False otherwise
    '''
    isMatchFound = False
    confidence = 0 
    
    if img_in is None:
      self.log.pLogErr("[match] Input image is empty!")
      return isMatchFound, confidence
    
    if template_in is None:
      self.log.pLogErr("[match] Template image is empty!")
      return isMatchFound, confidence

    # TODO Remove this exception handling and handle errors properly
    try:
      aligned_img = self.mtcnn(img_in)
      aligned_template = self.mtcnn(template_in)
    except:
      return isMatchFound, confidence
    
    # TODO Remove this exception handling and handle errors properly
    try:
      embeddings_img = self.resnet(aligned_img).detach()
      embeddings_template = self.resnet(aligned_template).detach()
    except:
      return isMatchFound, confidence
    
    # Compare the Euclidean distance between embeddings to matching threshold
    if embeddings_img is None or embeddings_template is None: 
      return isMatchFound, confidence
    
    try:
      if (((embeddings_img - embeddings_template).norm().item()) 
          < self.THR_EMBED_DISTANCE_FACE_MATCHING ):  
        isMatchFound = True
        confidence = ((embeddings_img - embeddings_template).norm().item()) 
    except:
      self.log.pLogWrn('[match] Computing euclidean distance failed.')
         
    return isMatchFound, confidence