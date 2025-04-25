import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *
from mediapipe.framework.formats import landmark_pb2
import time
import socket
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math
import queue
import os
import struct
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, TimeDistributed, ConvLSTM2D, BatchNormalization, MaxPooling3D
import tensorflow as tf
import collections
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define path to model directory
MODEL_DIR = "model_output/model_output"

# Load label map
try:
    with open(os.path.join(MODEL_DIR, 'label_map.json'), 'r') as f:
        ACTION_LABELS = json.load(f)
except Exception as e:
    print(f"Error loading label map: {e}")
    ACTION_LABELS = {"TennisSwing": 0, "BoxingPunchingBag": 1, "Diving": 2, "Archery": 3, 
                    "Basketball": 4, "LongJump": 5, "JugglingBalls": 6, "PushUps": 7, 
                    "BreastStroke": 8, "PullUps": 9}

# Function to build the model
def build_action_model(input_shape, num_classes):
    model = Sequential()
    
    # ConvLSTM blocks
    model.add(ConvLSTM2D(64, (3,3), activation='tanh', 
                        return_sequences=True,
                        input_shape=input_shape))
    model.add(MaxPooling3D((1,2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(ConvLSTM2D(128, (3,3), activation='tanh', return_sequences=True))
    model.add(MaxPooling3D((1,2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(ConvLSTM2D(256, (3,3), activation='tanh', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Classifier
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', 
                 optimizer='adam', 
                 metrics=['accuracy'])
    
    return model

# Advanced Pose Detector class for action recognition
class ActionClassifier:
    def __init__(self):
        # Model parameters
        self.frames_per_video = 30
        self.img_size = (60, 60)
        self.classes = list(ACTION_LABELS.keys())
        self.frame_buffer = collections.deque(maxlen=self.frames_per_video)
        
        # For smoothing predictions
        self.prediction_history = collections.deque(maxlen=5)
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # Update prediction every 1 second
        
        # Create and load the model
        try:
            print("\nInitializing action recognition model...")
            model_input_shape = (self.frames_per_video, self.img_size[0], self.img_size[1], 3)
            self.model = build_action_model(model_input_shape, len(self.classes))
            
            # Try to load pre-trained weights
            weights_file = os.path.join(MODEL_DIR, 'model_weights.weights.h5')
            model_file = os.path.join(MODEL_DIR, 'model.h5')
            
            if os.path.exists(model_file):
                print(f"Loading model from {model_file}...")
                try:
                    self.model = load_model(model_file)
                    print("Model loaded successfully!")
                except Exception as e:
                    print(f"Error loading model file: {e}")
                    if os.path.exists(weights_file):
                        print(f"Trying to load weights file instead...")
                        self.model.load_weights(weights_file)
                        print("Weights loaded successfully!")
            elif os.path.exists(weights_file):
                print(f"Loading weights from {weights_file}...")
                try:
                    self.model.load_weights(weights_file)
                    print("Weights loaded successfully!")
                except Exception as e:
                    print(f"Error loading weights: {e}")
            else:
                print("\nWARNING: No model or weights were found. The model will not produce accurate predictions.")
            
            print("Action recognition model ready for inference.")
            
        except Exception as e:
            print(f"\nError: Failed to build/load action recognition model: {e}")
            self.model = None
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame for the model."""
        # Resize the frame
        resized = cv2.resize(frame, self.img_size)
        # Convert to RGB (model was trained on RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize the frame (divide by 255 as in training)
        normalized = resized / 255.0
        return normalized
    
    def add_frame(self, frame):
        """Add a processed frame to the buffer."""
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
    
    def is_buffer_full(self):
        """Check if the frame buffer has enough frames."""
        return len(self.frame_buffer) == self.frames_per_video
    
    def get_majority_prediction(self):
        """Return the most common prediction from recent history."""
        if not self.prediction_history:
            return None, 0
        
        # Count occurrences of each class
        class_counts = {}
        for pred, conf in self.prediction_history:
            if pred in class_counts:
                class_counts[pred].append(conf)
            else:
                class_counts[pred] = [conf]
        
        # Find the most common class
        majority_class = max(class_counts.keys(), key=lambda k: len(class_counts[k]))
        avg_confidence = sum(class_counts[majority_class]) / len(class_counts[majority_class])
        
        return majority_class, avg_confidence * 100  # Convert confidence to percentage
    
    def predict(self, verbose=0):
        """Make a prediction based on the current frame buffer."""
        if not self.is_buffer_full() or self.model is None:
            return None, 0
        
        # Check if it's time for a new prediction
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            # Return the last stable prediction
            return self.get_majority_prediction()
        
        try:
            # Stack frames into a batch
            frames_array = np.array(list(self.frame_buffer))
            batch = np.expand_dims(frames_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(batch, verbose=verbose)
            
            # Get the predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Map index to class name (assuming self.classes is ordered as in the label map)
            predicted_class = self.classes[predicted_class_idx]
            
            # Add to prediction history
            self.prediction_history.append((predicted_class, confidence))
            self.last_prediction_time = current_time
            
            # Return smoothed prediction
            return self.get_majority_prediction()
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0

class PoseVisualizerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Pose Estimation")
        self.root.geometry("1920x1080")  # Full HD resolution
        
        # Initialize video capture with fallback
        self.current_camera = 0
        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            self.current_camera = 1 if self.current_camera == 0 else 0
            self.cap = cv2.VideoCapture(self.current_camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator()
        
        # Initialize action recognition model
        try:
            # Create instance of our advanced action classifier
            self.action_classifier = ActionClassifier()
            self.classification_result = None
            print("Action recognition model initialized successfully")
        except Exception as e:
            print(f"Error initializing action recognition model: {str(e)}")
            self.action_classifier = None
        
        # Initialize sensor data
        self.sensor_ip = None
        self.sensor_port = 8080
        self.sensor_data = {'x': [], 'y': [], 'z': [], 'time': []}
        self.max_sensor_points = 100
        
        # Initialize sensor analysis data
        self.sensor_analysis = {
            'gyro': {
                'x': [], 'y': [], 'z': [],
                'magnitude': [],
                'time': []
            },
            'accel': {
                'x': [], 'y': [], 'z': [],
                'magnitude': [],
                'time': []
            },
            'speed': [],
            'max_speed': 0,
            'avg_speed': 0,
            'max_gyro': 0,
            'avg_gyro': 0,
            'max_accel': 0,
            'avg_accel': 0
        }

    def update_action_classification_display(self):
        """Add action classification display below motion analysis"""
        try:
            # Create action classification overlay if it doesn't exist
            if not hasattr(self, 'action_classification_overlay'):
                self.action_classification_overlay = tk.Canvas(self.colored_frame, bg='black', highlightthickness=0)
                self.action_classification_overlay.place(x=self.colored_frame.winfo_width()-310, y=210, width=300, height=120)
            
            # Clear previous content
            self.action_classification_overlay.delete("all")
            
            # Draw background
            self.action_classification_overlay.create_rectangle(0, 0, 300, 120, fill='black', stipple='gray50')
            
            # Draw title with smaller font
            self.action_classification_overlay.create_text(150, 15, text="Action Classification", fill='white', font=('Helvetica', 10, 'bold'))
            
            # Draw current action and percentage if available
            if hasattr(self, 'classification_result') and self.classification_result:
                action, confidence = self.classification_result
                
                # Draw action name with larger font
                self.action_classification_overlay.create_text(150, 45, 
                    text=f"{action}", 
                    fill='white', font=('Helvetica', 14, 'bold'))
                
                # Draw confidence bar
                bar_width = 200
                bar_height = 20
                x_start = 50
                y_pos = 70
                
                # Background bar (gray)
                self.action_classification_overlay.create_rectangle(
                    x_start, y_pos,
                    x_start + bar_width, y_pos + bar_height,
                    fill='gray', outline='white')
                
                # Confidence bar (green gradient based on confidence)
                confidence_width = int(bar_width * confidence / 100)
                
                # Color gradient based on confidence
                if confidence < 30:
                    fill_color = 'red'
                elif confidence < 70:
                    fill_color = 'orange'
                else:
                    fill_color = 'green'
                    
                self.action_classification_overlay.create_rectangle(
                    x_start, y_pos,
                    x_start + confidence_width, y_pos + bar_height,
                    fill=fill_color, outline='')
                
                # Confidence text
                self.action_classification_overlay.create_text(150, 100, 
                    text=f"Confidence: {confidence:.1f}%", 
                    fill='white', font=('Helvetica', 10))
            else:
                # No action detected
                self.action_classification_overlay.create_text(150, 60, 
                    text="No Action Detected", 
                    fill='white', font=('Helvetica', 12))
        except Exception as e:
            print(f"Error updating action classification display: {e}")

    def classify_action(self, frame):
        """Classify action from the current frame using the 10-class classifier"""
        if self.action_classifier is None:
            return
            
        try:
            # Add frame to classifier's buffer
            self.action_classifier.add_frame(frame)
            
            # Get prediction if buffer is full
            if self.action_classifier.is_buffer_full():
                action, confidence = self.action_classifier.predict()
                if action is not None:
                    # Store classification result for display in the action classification overlay
                    self.classification_result = (action, confidence)
        except Exception as e:
            print(f"Error in action classification: {str(e)}")

    def update(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Process frame
                input_frame, skeleton_frame, colored_frame = self.pose_estimator.process_frame(frame)
                
                # Resize frames for each grid cell
                input_frame = cv2.resize(input_frame, (320, 240))  # Smaller input feed
                skeleton_frame = cv2.resize(skeleton_frame, (320, 240))  # Smaller 2D skeleton
                colored_frame = cv2.resize(colored_frame, (1280, 720))  # Larger 3D view
                
                # Convert frames to PhotoImage
                input_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)))
                skeleton_photo = ImageTk.PhotoImage(image=Image.fromarray(skeleton_frame))
                colored_photo = ImageTk.PhotoImage(image=Image.fromarray(colored_frame))
                
                # Update labels
                self.input_frame.configure(image=input_photo)
                self.skeleton_frame.configure(image=skeleton_photo)
                
                # Update canvas with 3D view
                self.colored_frame.delete("all")
                self.colored_frame.create_image(640, 360, image=colored_photo, anchor='center')
                
                # Keep references
                self.input_photo = input_photo
                self.skeleton_photo = skeleton_photo
                self.colored_photo = colored_photo
                
                # Update sensor data
                self.update_sensor_data()
                
                # Update motion analysis metrics
                if hasattr(self.pose_estimator.pose_3d, 'cosine_similarities') and self.pose_estimator.pose_3d.cosine_similarities:
                    self.motion_metrics['cosine_similarity'].configure(
                        text=f"Cosine Similarity: {self.pose_estimator.pose_3d.cosine_similarities[-1]:.3f}"
                    )
                
                if hasattr(self.pose_estimator.pose_3d, 'dtw_distances') and self.pose_estimator.pose_3d.dtw_distances:
                    self.motion_metrics['dtw_distance'].configure(
                        text=f"DTW Distance: {self.pose_estimator.pose_3d.dtw_distances[-1]:.3f}"
                    )
                
                # Process any pending mouse events
                self.root.update_idletasks()
                
                # Update motion analysis overlay
                self.update_motion_analysis_overlay()
                
                # Update motion analysis visualization if visible
                if self.motion_analysis_var.get() != "Data":
                    self.update_motion_analysis_visualization(self.motion_analysis_var.get())
                
                # Run action classification
                if self.action_classifier is not None:
                    self.classify_action(frame)
                
        except Exception as e:
            print(f"Error in update: {str(e)}")
        
        # Schedule next update with reduced delay
        self.root.after(5, self.update)

# ... rest of the code ...