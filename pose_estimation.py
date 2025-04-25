import os
import sys
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import threading
import json
import collections
import requests
# Define that requests is available since it was successfully imported
REQUESTS_AVAILABLE = True
import socket
import datetime
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import BytesIO

# Try to import TensorFlow components
try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Dense, Dropout, Flatten, Conv2D, MaxPooling2D, 
        ConvLSTM2D, BatchNormalization, Input, MaxPooling3D
    )
    from tensorflow.keras import Sequential
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Some features will be disabled.")
    TENSORFLOW_AVAILABLE = False

# Import pose 3D visualizer
from pose_3d import Pose3DVisualizer

# Try to import AI feedback module
try:
    from ai_feedback import GeminiCoach, SUPPORTED_EXERCISES, draw_coach_feedback
    AI_FEEDBACK_AVAILABLE = True
    print("AI Coach module loaded successfully")
except ImportError:
    AI_FEEDBACK_AVAILABLE = False
    print("AI Coach module not available. Coaching features will be disabled.")

# Sports Action Recognizer that uses the 10 activity model
class SportsActionRecognizer:
    def __init__(self, model_path='model_output/model_output/'):
        self.model_path = model_path
        self.frame_buffer = collections.deque(maxlen=10)
        self.prediction_history = collections.deque(maxlen=5)
        self.last_prediction_time = 0
        self.prediction_interval = 3.0  # Slow down to predict every 3 seconds
        
        # Load label map
        try:
            with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
                self.label_map = json.load(f)
            
            # Create reverse label map (index to label)
            self.classes = list(self.label_map.keys())
            self.idx_to_label = {i: label for i, label in enumerate(self.classes)}
            
            print(f"Sports Action Recognizer initialized with {len(self.classes)} classes")
            print(f"Classes: {self.classes}")
            
            # Build model from scratch with the exact architecture from training
            self.build_model()
            
            # Check for weights file
            weights_file = os.path.join(model_path, 'model_weights.weights.h5')
            if not os.path.exists(weights_file):
                weights_file = os.path.join(model_path, 'model.h5')
                
            if os.path.exists(weights_file):
                try:
                    print(f"Loading weights from {weights_file}...")
                    self.model.load_weights(weights_file, by_name=True, skip_mismatch=True)
                    print("Weights loaded successfully")
                    self.model_ready = True
                except Exception as e:
                    print(f"Error loading weights: {str(e)}")
                    self.model_ready = False
            else:
                print("Warning: Model weights file not found, using simulation mode")
                self.model_ready = False
            
        except Exception as e:
            print(f"Error initializing Sports Action Recognizer: {str(e)}")
            self.model_ready = False
            self.classes = ["PullUps", "PushUps", "JugglingBalls", "Boxing", "TennisSwing", 
                          "Diving", "Basketball", "Archery", "BreastStroke", "LongJump"]
            self.idx_to_label = {i: label for i, label in enumerate(self.classes)}
    
    def build_model(self):
        """Build the model EXACTLY as in the training code"""
        # Use exact same architecture as used in Kaggle
        input_shape = (10, 60, 60, 3)
        num_classes = len(self.classes)
        
        model = Sequential()
        
        # Input layer - explicitly use Input layer
        model.add(Input(shape=input_shape))
        
        # ConvLSTM blocks - exactly like in training code
        model.add(ConvLSTM2D(64, (3,3), activation='tanh', return_sequences=True))
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
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()  # Print summary to verify architecture
        
        self.model = model
        print("Model architecture built successfully")
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame for the model - exactly as in training."""
        # Resize to 60x60 pixels
        resized = cv2.resize(frame, (60, 60))
        # Convert to RGB (model was trained on RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize by dividing by 255
        normalized = resized / 255.0
        return normalized
    
    def add_frame(self, frame):
        """Add a processed frame to the buffer."""
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
    
    def is_buffer_full(self):
        """Check if the frame buffer has enough frames."""
        return len(self.frame_buffer) == 10  # Exactly 10 frames as in training
    
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
        
        return majority_class, avg_confidence * 100  # Convert to percentage
    
    def predict(self, frame):
        """Process a new frame and make a prediction if conditions are met."""
        # Add frame to buffer
        self.add_frame(frame)
            
        # Check if it's time for a new prediction and buffer is full
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval or not self.is_buffer_full():
            # Return the last stable prediction if we have one
            if len(self.prediction_history) > 0:
                return self.get_majority_prediction()
            else:
                # If model is not ready or we don't have predictions yet, use simulation
                if not self.model_ready:
                    predicted_class_idx = np.random.randint(0, len(self.classes))
                    confidence = np.random.uniform(0.7, 0.99)
                    predicted_class = self.classes[predicted_class_idx]
                    return predicted_class, confidence * 100
                return "Waiting...", 0
        
        try:
            if not self.model_ready:
                # If model not ready, use simulation mode
                predicted_class_idx = np.random.randint(0, len(self.classes))
                confidence = np.random.uniform(0.7, 0.99)
                predicted_class = self.classes[predicted_class_idx]
                
                # Add to prediction history
                self.prediction_history.append((predicted_class, confidence))
                self.last_prediction_time = current_time
                
                return predicted_class, confidence * 100
            
            # Stack frames into a batch - exactly as in training
            frames_array = np.array(list(self.frame_buffer))
            batch = np.expand_dims(frames_array, axis=0)  # Add batch dimension
            
            # Make prediction with the model
            predictions = self.model.predict(batch, verbose=0)
            
            # Get the predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Map index to class name
            predicted_class = self.classes[predicted_class_idx]
            
            # Add to prediction history
            self.prediction_history.append((predicted_class, confidence))
            self.last_prediction_time = current_time
            
            # Return smoothed prediction
            return self.get_majority_prediction()
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # If error, use last prediction if available
            if len(self.prediction_history) > 0:
                return self.get_majority_prediction()
            return None, 0

# Import our action recognition model
# from ml_models.action_recognition_model import ActionRecognitionModel

# Define a wrapper to integrate with our pose estimation
class ActionClassifier:
    def __init__(self):
        # Define paths to model and label map
        model_path = 'ml_models/action_recognition_model_enhanced.h5'
        label_map_path = 'class/label_map.json'
        
        # Fallback to model_new if enhanced model isn't available
        if not os.path.exists(model_path):
            model_path = 'ml_models/action_recognition_model_new.h5'
            print(f"Using alternative model: {model_path}")
        
        # Create hardcoded label map in case file can't be loaded
        self.hardcoded_label_map = {
            "PullUps": 0,
            "PushUps": 1,
            "JugglingBalls": 2,
            "BoxingSpeedBag": 3,
            "Punch": 4
        }
        
        # Define confidence thresholds for each action to reduce false positives
        self.confidence_thresholds = {
            "PullUps": 60.0,       # Base threshold
            "PushUps": 75.0,       # Higher threshold - reduce false positives
            "JugglingBalls": 80.0,  # Even higher threshold - reduce false positives
            "BoxingSpeedBag": 65.0, # Slightly higher than base
            "Punch": 65.0          # Slightly higher than base
        }
        
        # Keep track of previous classifications for smoothing
        self.prev_classifications = []
        self.max_prev_class = 5    # Number of previous classifications to remember
        self.stability_threshold = 3  # Minimum consecutive predictions to confirm action
        
        try:
            # Try to initialize the action recognition model
            self.model = ActionRecognitionModel(model_path, label_map_path)
            print(f"Action recognition model loaded from {model_path}")
            # Start in exercise mode for continuous classification
            self.model.is_exercising = True
        except Exception as e:
            print(f"Error initializing ActionRecognitionModel: {str(e)}")
            print("Falling back to simplified model...")
            self.model = None
            
    def process_frame(self, frame):
        """Process a single frame and return classification"""
        if self.model is None:
            # Return no result if model failed to load instead of simulating
            return "No Model", 0
            
        try:
            # Add frame to model buffer
            self.model.add_frame(frame)
            
            # Get classification
            if len(self.model.frame_buffer) >= self.model.buffer_size:
                action_name, confidence = self.model.classify_action()
                
                # Apply confidence thresholds
                if action_name and confidence < self.confidence_thresholds.get(action_name, 60.0):
                    # If below threshold, return "No Action"
                    return "No Action", confidence
                
                # Add to classification history
                self.prev_classifications.append(action_name)
                
                # Keep only last N classifications
                if len(self.prev_classifications) > self.max_prev_class:
                    self.prev_classifications.pop(0)
                
                # Check for stability - get the most common action in history
                if len(self.prev_classifications) >= self.stability_threshold:
                    # Count occurrences of each action
                    action_counts = {}
                    for a in self.prev_classifications:
                        action_counts[a] = action_counts.get(a, 0) + 1
                    
                    # Get most frequent action
                    most_frequent = max(action_counts.items(), key=lambda x: x[1])
                    most_frequent_action, count = most_frequent
                    
                    # If the most frequent action occurs at least stability_threshold times
                    if count >= self.stability_threshold:
                        return most_frequent_action, confidence
                
                return action_name, confidence
            else:
                # Not enough frames yet
                return None, 0
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return "Error", 0

class Pose3DVisualizer:
    def __init__(self, width=800, height=800):
        # Initialize Pygame without creating a window
        pygame.init()
        pygame.display.init()
        
        # Set OpenGL attributes
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        
        # Create a hidden window for OpenGL context
        self.display = (width, height)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | HIDDEN)
        
        # Initialize display parameters
        self.width = width
        self.height = height
        
        # Initialize camera parameters for better default view
        self.camera_distance = 5.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.rotation_x = 0
        self.rotation_y = 0  # Start with no rotation
        
        # Set up OpenGL
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width/height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Apply initial rotation to make the view correct
        glRotatef(90, 1, 0, 0)   # Rotate 90 degrees around X axis to make ground parallel to GUI
        glRotatef(180, 0, 1, 0)  # Rotate 180 degrees around Y axis to face forward
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 5, 5, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Initialize default pose
        self.default_pose = self.create_default_pose()
        self.current_pose = self.default_pose.copy()
        self.is_person_detected = False
        self.transition_speed = 0.1
        
        # Background color
        self.is_dark_mode = True
        self.bg_color_dark = (0.1, 0.1, 0.1, 1.0)
        self.bg_color_light = (0.9, 0.9, 0.9, 1.0)
        
        # Mouse control variables
        self.is_dragging = False
        self.last_x = 0
        self.last_y = 0
        
        # Add joint metrics tracking
        self.joint_metrics = {
            'left_shoulder': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 180}},
            'right_shoulder': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 180}},
            'left_elbow': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 145}},
            'right_elbow': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 145}},
            'left_hip': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 120}},
            'right_hip': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 120}},
            'left_knee': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 145}},
            'right_knee': {'angle': 0, 'speed': 0, 'range': {'min': 0, 'max': 145}}
        }
        
        # Add click detection
        self.selected_joint = None
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds
        
        # Add metrics display
        self.metrics_font = pygame.font.Font(None, 24)
        self.metrics_text = ""
        
        # Add colors for different body parts
        self.colors = {
            'torso': (0.8, 0.8, 0.8),  # Light gray
            'head': (0.9, 0.8, 0.7),   # Skin tone
            'left_arm': (0.7, 0.7, 0.9),
            'right_arm': (0.9, 0.7, 0.7),
            'left_leg': (0.7, 0.9, 0.7),
            'right_leg': (0.9, 0.7, 0.7)
        }

        # Add motion analysis attributes
        self.pose_history = []
        self.max_history = 100
        self.ideal_pose = None
        self.cosine_similarities = []
        self.dtw_distances = []
        
        # Add joint circle parameters
        self.joint_circle_radius = 0.05
        self.joint_circle_segments = 16

    def create_default_pose(self):
        """Create a default T-pose for the model with more detailed keypoints"""
        default_pose = []
        
        # Create 33 landmarks (MediaPipe pose format)
        for i in range(33):
            landmark = type('Landmark', (), {'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 1.0})()
            default_pose.append(landmark)
        
        # Head and face (more detailed)
        default_pose[0].x, default_pose[0].y, default_pose[0].z = 0.5, 0.8, 0.0  # Nose
        default_pose[1].x, default_pose[1].y, default_pose[1].z = 0.5, 0.85, 0.0  # Between eyes
        default_pose[2].x, default_pose[2].y, default_pose[2].z = 0.45, 0.85, 0.05  # Left eye
        default_pose[3].x, default_pose[3].y, default_pose[3].z = 0.55, 0.85, 0.05  # Right eye
        default_pose[4].x, default_pose[4].y, default_pose[4].z = 0.45, 0.87, 0.05  # Left ear
        default_pose[5].x, default_pose[5].y, default_pose[5].z = 0.55, 0.87, 0.05  # Right ear
        
        # Shoulders (wider stance)
        default_pose[11].x, default_pose[11].y, default_pose[11].z = 0.2, 0.65, 0.0  # Left shoulder
        default_pose[12].x, default_pose[12].y, default_pose[12].z = 0.8, 0.65, 0.0  # Right shoulder
        
        # Arms (more natural position)
        default_pose[13].x, default_pose[13].y, default_pose[13].z = 0.1, 0.65, 0.0  # Left elbow
        default_pose[14].x, default_pose[14].y, default_pose[14].z = 0.9, 0.65, 0.0  # Right elbow
        default_pose[15].x, default_pose[15].y, default_pose[15].z = 0.0, 0.65, 0.0  # Left wrist
        default_pose[16].x, default_pose[16].y, default_pose[16].z = 1.0, 0.65, 0.0  # Right wrist
        
        # Torso (more natural proportions)
        default_pose[23].x, default_pose[23].y, default_pose[23].z = 0.45, 0.4, 0.0  # Left hip
        default_pose[24].x, default_pose[24].y, default_pose[24].z = 0.55, 0.4, 0.0  # Right hip
        
        # Legs (wider stance)
        default_pose[25].x, default_pose[25].y, default_pose[25].z = 0.45, 0.25, 0.0  # Left knee
        default_pose[26].x, default_pose[26].y, default_pose[26].z = 0.55, 0.25, 0.0  # Right knee
        default_pose[27].x, default_pose[27].y, default_pose[27].z = 0.45, 0.1, 0.0  # Left ankle
        default_pose[28].x, default_pose[28].y, default_pose[28].z = 0.55, 0.1, 0.0  # Right ankle
        
        return default_pose

    def transform_coordinates(self, x, y, z):
        """Transform coordinates from MediaPipe space to our 3D space"""
        try:
            # Convert from MediaPipe coordinate system to OpenGL coordinate system
            x = np.clip((x - 0.5) * 2.0, -1.0, 1.0)
            y = np.clip(-(y - 0.5) * 2.0, -1.0, 1.0)  # Invert Y for correct orientation
            z = np.clip(-z * 2.0, -1.0, 1.0)
            
            # Scale and position the model
            scale = 1.5  # Increased scale for better visibility
            x *= scale
            y *= scale
            z *= scale
            
            # Center the model and lift it up
            y += 0.5  # Lift the model up from the ground
            
            return x, y, z
        except Exception as e:
            print(f"Error in transform_coordinates: {e}")
            return 0, 0, 0

    def draw_skeleton(self, landmarks):
        """Draw the skeleton using lines and joint circles"""
        if not landmarks:
            return

        try:
            # Transform all landmarks to 3D space
            points = {}
            for i, landmark in enumerate(landmarks):
                x, y, z = self.transform_coordinates(
                    landmark.x,
                    landmark.y,
                    landmark.z if hasattr(landmark, 'z') else 0.0
                )
                points[i] = np.array([x, y, z])

            # Draw connections
            glLineWidth(2.0)
            glBegin(GL_LINES)
            
            # Draw body parts in different colors
            # Torso - White
            glColor3f(1.0, 1.0, 1.0)
            self._draw_connection(points, [(11, 12), (12, 24), (24, 23), (23, 11)])
            
            # Left arm - Blue
            glColor3f(0.0, 0.0, 1.0)
            self._draw_connection(points, [(11, 13), (13, 15)])
            
            # Right arm - Red
            glColor3f(1.0, 0.0, 0.0)
            self._draw_connection(points, [(12, 14), (14, 16)])
            
            # Left leg - Yellow
            glColor3f(1.0, 1.0, 0.0)
            self._draw_connection(points, [(23, 25), (25, 27)])
            
            # Right leg - Green
            glColor3f(0.0, 1.0, 0.0)
            self._draw_connection(points, [(24, 26), (26, 28)])
            
            glEnd()
            glLineWidth(1.0)
            
            # Draw joint circles only for specific joints with dark red color
            glColor3f(0.8, 0.0, 0.0)  # Dark red color
            
            # Define the joints to circle
            circled_joints = [
                11, 12,  # Shoulders
                13, 14,  # Elbows
                15, 16,  # Wrists
                23, 24,  # Hips
                25, 26,  # Knees
                27, 28   # Ankles
            ]
            
            # Draw circles only for specified joints
            for joint_idx in circled_joints:
                if joint_idx in points:
                    self.draw_joint_circle(points[joint_idx])
                
        except Exception as e:
            print(f"Error in draw_skeleton: {e}")

    def calculate_joint_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def update_joint_metrics(self, landmarks):
        """Update joint metrics based on current pose"""
        if not landmarks:
            return
            
        # Transform landmarks to 3D space
        points = {}
        for i, landmark in enumerate(landmarks):
            x, y, z = self.transform_coordinates(
                landmark.x,
                landmark.y,
                landmark.z if hasattr(landmark, 'z') else 0.0
            )
            points[i] = np.array([x, y, z])
        
        # Update metrics for each joint
        current_time = time.time()
        
        # Left shoulder
        if all(k in points for k in [11, 13, 15]):
            angle = self.calculate_joint_angle(points[11], points[13], points[15])
            self.joint_metrics['left_shoulder']['speed'] = abs(angle - self.joint_metrics['left_shoulder']['angle']) / (current_time - self.last_click_time)
            self.joint_metrics['left_shoulder']['angle'] = angle
            
        # Right shoulder
        if all(k in points for k in [12, 14, 16]):
            angle = self.calculate_joint_angle(points[12], points[14], points[16])
            self.joint_metrics['right_shoulder']['speed'] = abs(angle - self.joint_metrics['right_shoulder']['angle']) / (current_time - self.last_click_time)
            self.joint_metrics['right_shoulder']['angle'] = angle
            
        # Similar updates for other joints...
        
        self.last_click_time = current_time

    def handle_mouse_event(self, event_type, event):
        """Handle mouse events for 3D controls"""
        if event_type == 'mousedown':
            self.is_dragging = True
            self.last_x = event.x
            self.last_y = event.y
        elif event_type == 'mouseup':
            self.is_dragging = False
        elif event_type == 'mousemotion' and self.is_dragging:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            
            # Left mouse button for both rotation and panning
            if event.state & 0x0001:  # Left mouse button
                # Use vertical movement for rotation
                self.rotation_x += dy * 0.3
                # Use horizontal movement for panning
                self.camera_x += dx * 0.01
                
            self.last_x = event.x
            self.last_y = event.y
        elif event_type == 'mousewheel':
            if event.delta > 0:
                self.camera_distance = max(2, self.camera_distance - 0.3)
            else:
                self.camera_distance = min(10, self.camera_distance + 0.3)

    def render_pose(self, landmarks=None):
        """Render the 3D pose with metrics and motion analysis"""
        try:
            # Clear the screen and depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(*self.bg_color_dark if self.is_dark_mode else self.bg_color_light)
            
            # Reset modelview matrix
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Apply camera transformations
            glTranslatef(self.camera_x, self.camera_y, -self.camera_distance)
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 1, 0)
            
            # Apply initial rotation to make the view correct
            glRotatef(90, 1, 0, 0)   # Rotate 90 degrees around X axis to make ground parallel to GUI
            glRotatef(180, 0, 1, 0)  # Rotate 180 degrees around Y axis to face forward
            
            # Draw reference grid
            self.draw_grid()
            
            # Draw platform
            self.draw_platform()
            
            # Update and draw pose
            if landmarks and any(hasattr(lm, 'visibility') and lm.visibility > 0.5 for lm in landmarks):
                self.is_person_detected = True
                self.current_pose = self.lerp_poses(self.current_pose, landmarks, self.transition_speed)
            else:
                self.is_person_detected = False
                self.current_pose = self.lerp_poses(self.current_pose, self.default_pose, 0.2)
            
            # Draw the body silhouette first
            self.draw_body_silhouette(self.current_pose)
            
            # Draw the skeleton on top
            self.draw_skeleton(self.current_pose)
            
            # Update joint metrics
            self.update_joint_metrics(landmarks)
            
            # Draw metrics
            self.draw_metrics()
            
            # Update motion analysis
            self.update_motion_analysis(landmarks)
            
            # Draw motion analysis overlay
            self.draw_motion_analysis_overlay()
            
            # Get the rendered image
            buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(buffer, dtype=np.uint8)
            image = image.reshape((self.height, self.width, 3))
            image = np.flipud(image)
            
            # Convert to Pygame surface
            self.screen = pygame.surfarray.make_surface(image)
            
        except Exception as e:
            print(f"Error in render_pose: {e}")

    def draw_grid(self):
        """Draw a reference grid"""
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i/2, -1, -5)
            glVertex3f(i/2, -1, 5)
            glVertex3f(-5, -1, i/2)
            glVertex3f(5, -1, i/2)
        glEnd()

    def draw_platform(self):
        """Draw a simple platform"""
        try:
            glColor3f(0.4, 0.4, 0.4)
            glBegin(GL_POLYGON)
            num_segments = 32
            radius = 1.0  # Increased radius for bigger platform
            for i in range(num_segments):
                theta = 2.0 * 3.1415926 * i / num_segments
                glVertex3f(radius * np.cos(theta), -0.99, radius * np.sin(theta))
            glEnd()
            
            # Draw platform label using lines
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_LINES)
            # Draw "P"
            glVertex3f(-0.8, -1.1, 0)
            glVertex3f(-0.8, -0.9, 0)
            glVertex3f(-0.8, -1.0, 0)
            glVertex3f(-0.6, -1.0, 0)
            glVertex3f(-0.6, -1.1, 0)
            glVertex3f(-0.6, -0.9, 0)
            # Draw "L"
            glVertex3f(-0.4, -1.1, 0)
            glVertex3f(-0.4, -0.9, 0)
            glVertex3f(-0.4, -1.1, 0)
            glVertex3f(-0.2, -1.1, 0)
            # Draw "A"
            glVertex3f(0.0, -1.1, 0)
            glVertex3f(0.0, -0.9, 0)
            glVertex3f(0.0, -1.0, 0)
            glVertex3f(0.2, -1.0, 0)
            glVertex3f(0.2, -1.1, 0)
            glVertex3f(0.2, -0.9, 0)
            # Draw "T"
            glVertex3f(0.4, -0.9, 0)
            glVertex3f(0.6, -0.9, 0)
            glVertex3f(0.5, -0.9, 0)
            glVertex3f(0.5, -1.1, 0)
            # Draw "F"
            glVertex3f(0.8, -1.1, 0)
            glVertex3f(0.8, -0.9, 0)
            glVertex3f(0.8, -1.0, 0)
            glVertex3f(1.0, -1.0, 0)
            glVertex3f(0.8, -1.1, 0)
            glVertex3f(1.0, -1.1, 0)
            glEnd()
        except Exception as e:
            print(f"Error in draw_platform: {e}")

    def _draw_connection(self, points, connections):
        """Helper function to draw connections between points"""
        for start_idx, end_idx in connections:
            if start_idx in points and end_idx in points:
                glVertex3f(*points[start_idx])
                glVertex3f(*points[end_idx])

    def draw_metrics(self):
        """Draw joint metrics on screen"""
        if self.metrics_text:
            lines = self.metrics_text.split('\n')
            y = 20
            for line in lines:
                text_surface = self.metrics_font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, y))
                y += 25

    def resize(self, width, height):
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width/height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def draw_cylinder(self, start, end, radius=0.05):
        """Draw a 3D cylinder between two points"""
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        
        if length < 0.0001:
            return
            
        direction = direction / length
        
        # Create rotation matrix
        up = np.array([0, 1, 0])
        if abs(np.dot(up, direction)) > 0.99:
            up = np.array([1, 0, 0])
        
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)
        
        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = direction
        transform[:3, 2] = up
        transform[:3, 3] = start
        
        glPushMatrix()
        glMultMatrixf(transform.T)
        
        quad = gluNewQuadric()
        gluCylinder(quad, radius, radius, length, 10, 5)
        
        glPopMatrix()

    def draw_sphere(self, position, radius=0.08):
        """Draw a sphere at joint position"""
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        quad = gluNewQuadric()
        gluSphere(quad, radius, 10, 10)
        glPopMatrix()

    def toggle_background(self):
        """Toggle between light and dark background"""
        self.is_dark_mode = not self.is_dark_mode

    def lerp_poses(self, start_pose, end_pose, t):
        """Linearly interpolate between two poses"""
        result = []
        for i in range(len(start_pose)):
            landmark = type('Landmark', (), {
                'x': start_pose[i].x + (end_pose[i].x - start_pose[i].x) * t,
                'y': start_pose[i].y + (end_pose[i].y - start_pose[i].y) * t,
                'z': start_pose[i].z + (end_pose[i].z - start_pose[i].z) * t,
                'visibility': 1.0
            })()
            result.append(landmark)
        return result

    def draw_body_silhouette(self, landmarks):
        """Draw a basic body silhouette around the skeleton"""
        if not landmarks:
            return

        # Transform all landmarks to 3D space
        points = {}
        for i, landmark in enumerate(landmarks):
            x, y, z = self.transform_coordinates(
                landmark.x,
                landmark.y,
                landmark.z if hasattr(landmark, 'z') else 0.0
            )
            points[i] = np.array([x, y, z])

        # Draw filled polygons for each segment
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Skin color with transparency
        glColor4f(0.94, 0.75, 0.64, 0.5)
        
        # Draw torso as a solid shape
        if all(k in points for k in [11, 12, 23, 24]):
            glBegin(GL_QUADS)
            glVertex3f(*points[11])
            glVertex3f(*points[12])
            glVertex3f(*points[24])
            glVertex3f(*points[23])
            glEnd()
        
        # Draw limbs with thickness
        segments = {
            'arms': [(11, 13, 15), (12, 14, 16)],
            'legs': [(23, 25, 27), (24, 26, 28)]
        }
        
        for segment_type, segment_list in segments.items():
            for segment in segment_list:
                for i in range(len(segment)-1):
                    if segment[i] in points and segment[i+1] in points:
                        start = points[segment[i]]
                        end = points[segment[i+1]]
                        
                        # Calculate thickness based on segment type
                        thickness = 0.08 if segment_type == 'arms' else 0.1
                        
                        # Draw thick line
                        direction = end - start
                        length = np.linalg.norm(direction)
                        if length > 0:
                            direction = direction / length
                            perpendicular = np.cross(direction, np.array([0, 1, 0]))
                            perpendicular = perpendicular / np.linalg.norm(perpendicular)
                            
                            glBegin(GL_QUADS)
                            glVertex3f(*(start + perpendicular * thickness))
                            glVertex3f(*(start - perpendicular * thickness))
                            glVertex3f(*(end - perpendicular * thickness))
                            glVertex3f(*(end + perpendicular * thickness))
                            glEnd()
        
        # Draw head
        if 0 in points:
            glPushMatrix()
            glTranslatef(*points[0])
            sphere = gluNewQuadric()
            gluSphere(sphere, 0.15, 16, 16)
            glPopMatrix()
        
        glDisable(GL_BLEND)

    def apply_physics(self, landmarks):
        """Apply physics-based motion to landmarks"""
        if not landmarks:
            return landmarks
        
        result = []
        points = []
        
        # Convert landmarks to points
        for i, lm in enumerate(landmarks):
            x, y, z = self.transform_coordinates(lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0)
            points.append(np.array([x, y, z]))
        
        # Apply physics to each point
        for i in range(len(points)):
            # Calculate spring forces from connected joints
            force = np.zeros(3)
            for connections in self.body_segments.values():
                for start_idx, end_idx in connections:
                    if i == start_idx or i == end_idx:
                        other_idx = end_idx if i == start_idx else start_idx
                        if other_idx < len(points):
                            direction = points[other_idx] - points[i]
                            distance = np.linalg.norm(direction)
                            if distance > 0:
                                # Spring force
                                force += self.spring_constant * direction
            
            # Apply forces
            self.acceleration[i] = force / self.mass
            self.velocity[i] = self.damping * (self.velocity[i] + self.acceleration[i] * self.dt)
            new_pos = points[i] + self.velocity[i] * self.dt
            
            # Convert back to landmark
            x = (new_pos[0] / 1.5 + 0.5)
            y = (-new_pos[1] / 1.5 + 0.5)
            z = -new_pos[2] / 2.0
            
            landmark = type('Landmark', (), {'x': x, 'y': y, 'z': z, 'visibility': 1.0})()
            result.append(landmark)
        
        return result

    def calculate_cosine_similarity(self, pose1, pose2):
        """Calculate cosine similarity between two poses"""
        try:
            # Convert poses to vectors
            vec1 = np.array([lm.x for lm in pose1])
            vec2 = np.array([lm.x for lm in pose2])
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return similarity
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def calculate_dtw_distance(self, pose1, pose2):
        """Calculate Dynamic Time Warping distance between poses"""
        try:
            # Convert poses to feature vectors
            vec1 = np.array([lm.x for lm in pose1])
            vec2 = np.array([lm.x for lm in pose2])
            
            # Calculate DTW distance
            n, m = len(vec1), len(vec2)
            dtw_matrix = np.zeros((n+1, m+1))
            dtw_matrix.fill(np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = abs(vec1[i-1] - vec2[j-1])
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                                dtw_matrix[i, j-1],
                                                dtw_matrix[i-1, j-1])
            
            return dtw_matrix[n, m]
        except Exception as e:
            print(f"Error calculating DTW distance: {e}")
            return 0.0

    def draw_joint_circle(self, position, radius=None):
        """Draw a circle around a joint"""
        if radius is None:
            radius = self.joint_circle_radius
            
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        
        # Draw circle in XY plane
        glBegin(GL_LINE_LOOP)
        for i in range(self.joint_circle_segments):
            theta = 2.0 * np.pi * i / self.joint_circle_segments
            glVertex3f(radius * np.cos(theta), radius * np.sin(theta), 0)
        glEnd()
        
        # Draw circle in XZ plane
        glBegin(GL_LINE_LOOP)
        for i in range(self.joint_circle_segments):
            theta = 2.0 * np.pi * i / self.joint_circle_segments
            glVertex3f(radius * np.cos(theta), 0, radius * np.sin(theta))
        glEnd()
        
        # Draw circle in YZ plane
        glBegin(GL_LINE_LOOP)
        for i in range(self.joint_circle_segments):
            theta = 2.0 * np.pi * i / self.joint_circle_segments
            glVertex3f(0, radius * np.cos(theta), radius * np.sin(theta))
        glEnd()
        
        glPopMatrix()

    def update_motion_analysis(self, landmarks):
        """Update motion analysis with new pose data"""
        try:
            if landmarks:
                # Add current pose to history
                self.pose_history.append(landmarks)
                if len(self.pose_history) > self.max_history:
                    self.pose_history.pop(0)
                
                # Calculate cosine similarity with previous pose
                if len(self.pose_history) > 1:
                    similarity = self.calculate_cosine_similarity(
                        self.pose_history[-1],
                        self.pose_history[-2]
                    )
                    self.cosine_similarities.append(similarity)
                    
                    # Keep only last 100 similarities
                    if len(self.cosine_similarities) > 100:
                        self.cosine_similarities.pop(0)
                
                # Calculate DTW distance from ideal pose if available
                if self.ideal_pose:
                    dtw_distance = self.calculate_dtw_distance(
                        landmarks,
                        self.ideal_pose
                    )
                    self.dtw_distances.append(dtw_distance)
                    
                    # Keep only last 100 distances
                    if len(self.dtw_distances) > 100:
                        self.dtw_distances.pop(0)
                
                # Generate spectrogram data
                if len(self.cosine_similarities) > 10:
                    # Convert similarities to numpy array
                    similarities_array = np.array(self.cosine_similarities)
                    
                    # Apply FFT
                    fft_data = np.fft.fft(similarities_array)
                    frequencies = np.fft.fftfreq(len(similarities_array))
                    
                    # Store frequency data for visualization
                    self.frequency_data = {
                        'frequencies': frequencies,
                        'magnitudes': np.abs(fft_data)
                    }
                    
        except Exception as e:
            print(f"Error in update_motion_analysis: {e}")

    def draw_motion_analysis_overlay(self):
        """Draw motion analysis overlay"""
        try:
            if hasattr(self, 'frequency_data'):
                # Draw frequency spectrum
                glPushMatrix()
                glTranslatef(-2.0, 2.0, -2.0)
                glColor3f(1.0, 1.0, 1.0)
                
                # Draw frequency spectrum as lines
                glBegin(GL_LINE_STRIP)
                for i in range(len(self.frequency_data['frequencies'])):
                    x = i / len(self.frequency_data['frequencies'])
                    y = self.frequency_data['magnitudes'][i] / max(self.frequency_data['magnitudes'])
                    glVertex3f(x, y, 0)
                glEnd()
                
                glPopMatrix()
                
        except Exception as e:
            print(f"Error in draw_motion_analysis_overlay: {e}")

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Reduced for better performance
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        # Initialize 3D visualizer with larger window
        self.pose_3d = Pose3DVisualizer(width=800, height=600)
        
        # Define body segments using MediaPipe's POSE_CONNECTIONS
        self.body_segments = {
            'head': [
                (0, 1), (1, 4), (4, 7),
                (0, 2), (2, 5), (5, 8),
                (1, 2), (4, 5), (7, 8)
            ],
            'torso': [
                (11, 12),
                (12, 24), (24, 23), (23, 11),
                (11, 23), (12, 24),
                (11, 13), (12, 14),
                (23, 25), (24, 26)
            ],
            'left_arm': [
                (11, 13), (13, 15)
            ],
            'right_arm': [
                (12, 14), (14, 16)
            ],
            'left_leg': [
                (23, 25), (25, 27)
            ],
            'right_leg': [
                (24, 26), (26, 28)
            ]
        }
        self.pose_3d.body_segments = self.body_segments

    def process_frame(self, frame):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        # Create visualization frame with landmarks drawn
        output_frame = frame.copy()
        landmarks = None
        landmark_world = None
        
        if results.pose_landmarks:
            # Draw skeleton on input frame
            self.mp_draw.draw_landmarks(
                output_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2))
            
            # Transform landmarks for 3D visualization
            landmarks = results.pose_landmarks.landmark
            landmark_world = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
            
            # Render the 3D pose if needed
            landmarks_3d = []
            for landmark in landmarks:
                # Map coordinates to our 3D space
                x = landmark.x
                y = landmark.y
                z = landmark.z if hasattr(landmark, 'z') else 0.0
                
                # Create transformed landmark
                transformed_landmark = type('Landmark', (), {
                    'x': x,
                    'y': y,
                    'z': z,
                    'visibility': landmark.visibility
                })()
                landmarks_3d.append(transformed_landmark)
            
            # Update 3D pose for separate rendering
            self.pose_3d.render_pose(landmarks_3d)
        else:
            # Render default T-pose when no person is detected
            self.pose_3d.render_pose(None)
        
        return output_frame, landmarks, landmark_world

class PoseVisualizerGUI:
    def __init__(self, cap=None):
        # Initialize pygame for 3D rendering
        if not pygame.get_init():
            pygame.init()
        
        # Store the video capture if provided
        if cap is not None:
            self.cap = cap
        
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Pose Visualizer")
        
        # Initialize mouse state for 3D view interaction
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Create main frame with 2x3 grid
        self.main_frame = ttk.Frame(self.root)
        # Replace pack with grid
        self.main_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=10, pady=10)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator()
        
        # Initialize exercise metrics and AI coach
        self.exercise_metrics = {
            'rep_count': 0,
            'form_score': 0,
            'speed': 0,
            'range_of_motion': 0,
            'stability': 0
        }
        self.exercise_active = False
        self.timer_start = None
        self.timers = []
        
        # Initialize sensor data
        self.sensor_connected = False
        self.sensor_data = {
            'gyro': {'x': [], 'y': [], 'z': [], 'magnitude': [], 'time': []},
            'accel': {'x': [], 'y': [], 'z': [], 'magnitude': [], 'time': []},
            'max_gyro': 0,
            'avg_gyro': 0,
            'max_accel': 0,
            'avg_accel': 0
        }
        
        # Initialize AI Coach if available
        if AI_FEEDBACK_AVAILABLE:
            try:
                self.ai_coach = GeminiCoach()
                self.coach_available = True
                self.current_feedback = "Start an exercise to get AI feedback"
                self.feedback_visible = True
                self.last_feedback_time = 0
                print("Gemini AI coach initialized successfully")
            except Exception as e:
                print(f"Error initializing AI coach: {e}")
                self.coach_available = False
                self.ai_coach = None
        else:
            self.coach_available = False
            self.ai_coach = None
            
        # Initialize sensor connection properties
        self.sensor_ip = None
        self.sensor_port = None
        self.last_known_ip = None
        self.last_known_port = None
        self.connection_retry_count = 0
        self.max_retries = 3
        self.sensor_check_interval = 1.0
        self.last_sensor_check = 0
        self.auto_reconnect = True
        
        # Configure grid weights for 2x3 layout - give more weight to middle and right columns
        self.main_frame.columnconfigure(0, weight=1)  # Left column (smaller)
        self.main_frame.columnconfigure(1, weight=3)  # Middle column (larger)
        self.main_frame.columnconfigure(2, weight=3)  # Right column (larger)
        self.main_frame.rowconfigure(0, weight=3)  # Top row (larger)
        self.main_frame.rowconfigure(1, weight=1)  # Bottom row (smaller)
        
        # Create a frame for the left column which will contain input feed, 2D skeleton, and future work space
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(column=0, row=0, rowspan=2, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        
        # Configure the left frame to have 3 rows - input feed, 2D skeleton, and future work space
        self.left_frame.rowconfigure(0, weight=1)  # Input feed
        self.left_frame.rowconfigure(1, weight=1)  # 2D skeleton
        self.left_frame.rowconfigure(2, weight=1)  # Future work space
        self.left_frame.columnconfigure(0, weight=1)
        
        # 1. Input Feed Canvas (top part of left frame) - 3x2 inches (approx 288x192 pixels)
        self.video_frame = ttk.LabelFrame(self.left_frame, text="Input Feed")
        self.video_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        self.video_canvas = tk.Canvas(self.video_frame, width=288, height=192, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 2. 2D Skeleton Canvas (middle part of left frame) - with black background
        self.skeleton_frame = ttk.LabelFrame(self.left_frame, text="2D Skeleton")
        self.skeleton_frame.grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        self.skeleton_canvas = tk.Canvas(self.skeleton_frame, width=288, height=192, bg="black")
        self.skeleton_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 3. Sensor Data 3D Visualization (bottom part of left frame) - renamed from Future Work
        self.sensor_3d_frame = ttk.LabelFrame(self.left_frame, text="Sensor Data 3D Visualization")
        self.sensor_3d_frame.grid(row=2, column=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        self.sensor_3d_canvas = tk.Canvas(self.sensor_3d_frame, width=288, height=192, bg="black")
        self.sensor_3d_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 4. 3D Visualization (middle and right top)
        self.visualization_frame = ttk.LabelFrame(self.main_frame, text="3D Visualization")
        self.visualization_frame.grid(column=1, row=0, columnspan=2, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        self.colored_frame = tk.Canvas(self.visualization_frame, bg='black', width=int(1280 + 192), height=720)
        self.colored_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure mouse events for 3D visualization
        self.colored_frame.bind("<ButtonPress-1>", self.on_mouse_down)
        self.colored_frame.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.colored_frame.bind("<B1-Motion>", self.on_mouse_drag)
        self.colored_frame.bind("<MouseWheel>", self.on_mouse_wheel)
        
        # Add support for right-click and middle-click dragging
        self.colored_frame.bind("<ButtonPress-2>", self.on_middle_mouse_down)
        self.colored_frame.bind("<B2-Motion>", self.on_middle_mouse_drag)
        self.colored_frame.bind("<ButtonPress-3>", self.on_right_mouse_down)
        self.colored_frame.bind("<B3-Motion>", self.on_right_mouse_drag)
        
        # Add keyboard navigation for 3D view
        self.root.bind("<KeyPress>", self.on_key_press)
        # Give the 3D view focus so it can receive key events
        self.colored_frame.focus_set()
        
        # 5. Sensor Data (middle bottom)
        self.sensor_frame = ttk.LabelFrame(self.main_frame, text="Sensor Data")
        self.sensor_frame.grid(column=1, row=1, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        
        self.sensor_canvas = tk.Canvas(self.sensor_frame, bg='black', width=400, height=200)
        self.sensor_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add connection status indicator
        self.connection_status = tk.Canvas(self.sensor_canvas, width=20, height=20, bg='black', highlightthickness=0)
        self.connection_status.place(x=10, y=10)
        self.connection_status.create_oval(0, 0, 20, 20, fill='red')
        
        # Add sensor type selector
        self.sensor_type = tk.StringVar(value="gyro")
        ttk.Radiobutton(self.sensor_canvas, text="Gyroscope", variable=self.sensor_type, value="gyro").place(x=40, y=10)
        ttk.Radiobutton(self.sensor_canvas, text="Accelerometer", variable=self.sensor_type, value="accel").place(x=150, y=10)
        
        # 6. Controls (right bottom)
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.grid(column=2, row=1, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        
        # Create scrollable control area
        self.control_canvas = tk.Canvas(self.control_frame)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.control_scrollbar = ttk.Scrollbar(self.control_frame, orient=tk.VERTICAL, command=self.control_canvas.yview)
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.bind('<Configure>', self.on_canvas_configure)
        
        self.controls = ttk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.controls, anchor=tk.NW)
        
        # Add controls here
        style = ttk.Style()
        style.configure('Modern.TButton', padding=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(self.controls, text="Camera Controls")
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(camera_frame, text="Switch Camera", style='Modern.TButton', command=self.switch_camera).pack(fill=tk.X, padx=5, pady=2)
        
        # Action Classification controls
        action_control_frame = ttk.LabelFrame(self.controls, text="Action Classification Controls")
        action_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Detection rate slider
        rate_frame = ttk.Frame(action_control_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Detection Rate (s):").pack(side=tk.LEFT, padx=5)
        self.detection_rate_var = tk.DoubleVar(value=3.0)  # Default 3.0 seconds
        rate_scale = ttk.Scale(rate_frame, from_=0.5, to=10.0, 
                               variable=self.detection_rate_var, 
                               command=self.update_detection_rate)
        rate_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.detection_rate_label = ttk.Label(rate_frame, text="3.0")
        self.detection_rate_label.pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold slider
        threshold_frame = ttk.Frame(action_control_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text="Confidence Threshold (%):").pack(side=tk.LEFT, padx=5)
        self.confidence_threshold_var = tk.DoubleVar(value=20.0)  # Default 20%
        threshold_scale = ttk.Scale(threshold_frame, from_=5.0, to=95.0, 
                                   variable=self.confidence_threshold_var, 
                                   command=self.update_confidence_threshold)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.threshold_label = ttk.Label(threshold_frame, text="20.0")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Detection control buttons
        buttons_frame = ttk.Frame(action_control_frame)
        buttons_frame.pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Faster Detection", 
                  style='Modern.TButton', 
                  command=self.increase_detection_rate).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(buttons_frame, text="Slower Detection", 
                  style='Modern.TButton', 
                  command=self.decrease_detection_rate).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # View controls
        view_frame = ttk.LabelFrame(self.controls, text="View Controls")
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(view_frame, text="Toggle Background", style='Modern.TButton', command=self.toggle_background).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(view_frame, text="Reset View", style='Modern.TButton', command=self.reset_view).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(view_frame, text="Rotate 90°", style='Modern.TButton', command=self.rotate_90_degrees).pack(fill=tk.X, padx=5, pady=2)
        
        # Sensor controls
        sensor_control_frame = ttk.LabelFrame(self.controls, text="Sensor Controls")
        sensor_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # IP and port inputs
        ip_frame = ttk.Frame(sensor_control_frame)
        ip_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ip_frame, text="IP:").pack(side=tk.LEFT, padx=5)
        self.ip_entry = ttk.Entry(ip_frame)
        self.ip_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.ip_entry.insert(0, "192.168.29.217")
        
        port_frame = ttk.Frame(sensor_control_frame)
        port_frame.pack(fill=tk.X, pady=2)
        ttk.Label(port_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = ttk.Entry(port_frame)
        self.port_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.port_entry.insert(0, "8080")
        
        # Connection buttons
        connect_frame = ttk.Frame(sensor_control_frame)
        connect_frame.pack(fill=tk.X, pady=2)
        ttk.Button(connect_frame, text="Connect", style='Modern.TButton', command=self.connect_sensor).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(connect_frame, text="Reconnect", style='Modern.TButton', command=self.reconnect_sensor).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Advanced sensor settings
        advanced_frame = ttk.LabelFrame(sensor_control_frame, text="Advanced Settings")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Auto reconnect checkbox
        reconnect_frame = ttk.Frame(advanced_frame)
        reconnect_frame.pack(fill=tk.X, pady=2)
        self.auto_reconnect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(reconnect_frame, text="Auto Reconnect", 
                     variable=self.auto_reconnect_var, 
                     command=self.update_reconnect_setting).pack(side=tk.LEFT, padx=5)
        
        # Timeout slider
        timeout_frame = ttk.Frame(advanced_frame)
        timeout_frame.pack(fill=tk.X, pady=2)
        ttk.Label(timeout_frame, text="Timeout (s):").pack(side=tk.LEFT, padx=5)
        self.timeout_var = tk.DoubleVar(value=0.5)
        timeout_scale = ttk.Scale(timeout_frame, from_=0.1, to=5.0, 
                               variable=self.timeout_var, 
                               command=self.update_timeout)
        timeout_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.timeout_label = ttk.Label(timeout_frame, text="0.5")
        self.timeout_label.pack(side=tk.LEFT, padx=5)
        
        # Update interval slider
        interval_frame = ttk.Frame(advanced_frame)
        interval_frame.pack(fill=tk.X, pady=2)
        ttk.Label(interval_frame, text="Update Interval (s):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.DoubleVar(value=1.0)
        interval_scale = ttk.Scale(interval_frame, from_=0.1, to=5.0, 
                                variable=self.interval_var, 
                                command=self.update_check_interval)
        interval_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.interval_label = ttk.Label(interval_frame, text="1.0")
        self.interval_label.pack(side=tk.LEFT, padx=5)
        
        # Max retries setting
        retries_frame = ttk.Frame(advanced_frame)
        retries_frame.pack(fill=tk.X, pady=2)
        ttk.Label(retries_frame, text="Max Retries:").pack(side=tk.LEFT, padx=5)
        self.max_retries_var = tk.IntVar(value=3)
        max_retries_spinbox = ttk.Spinbox(retries_frame, from_=1, to=10, 
                                       textvariable=self.max_retries_var,
                                       width=5,
                                       command=self.update_max_retries)
        max_retries_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Sensor data points to display
        data_points_frame = ttk.Frame(advanced_frame)
        data_points_frame.pack(fill=tk.X, pady=2)
        ttk.Label(data_points_frame, text="Display Points:").pack(side=tk.LEFT, padx=5)
        self.data_points_var = tk.IntVar(value=100)
        data_points_spinbox = ttk.Spinbox(data_points_frame, from_=10, to=500, 
                                       textvariable=self.data_points_var,
                                       width=5,
                                       command=self.update_data_points)
        data_points_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Initialize video capture with fallback
        if cap is None:
            self.current_camera = 0  # Track current camera index
            # Try multiple camera indices
            for camera_index in [0, 1, 2, 3]:
                print(f"Trying camera index {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    self.current_camera = camera_index  # Store the working camera index
                    print(f"Successfully opened camera {camera_index}")
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            
            if not self.cap.isOpened():
                self.cap = None
                tk.messagebox.showerror("Camera Error", "Could not open any camera. Please check your camera connection.")
        else:
            self.cap = cap
            self.current_camera = 0  # Default to 0 for provided camera
        
        # Check if camera is available, if not show an error
        if not self.cap.isOpened():
            self.cap = None
            tk.messagebox.showerror("Camera Error", "Could not open any camera. Please check your camera connection.")
        
        # Initialize PoseEstimator for 2D pose detection
        self.pose_estimator = PoseEstimator()
        
        # Initialize sensor data
        self.sensor_connected = False
        self.sensor_data = {
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
            'avg_accel': 0,
            'total_rotation': {'x': 0, 'y': 0, 'z': 0}
        }
        
        # Initialize motion analysis variables
        self.motion_analysis_var = tk.StringVar(value="Data")
        self.spectral_data = []
        
        # Initialize overlays for the 3D visualization
        self.analysis_overlay = None
        self.motion_analysis_overlay = None
        
        # Initialize action classifier
        try:
            self.action_classifier = PoseActionClassifier()
            print("PoseActionClassifier initialized successfully with model")
        except Exception as e:
            print(f"Failed to initialize action classifier: {e}")
            self.action_classifier = None
        
        # Mouse state tracking for 3D view
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Add Action Classification controls and AI Coaching to the control frame
        action_control_frame = ttk.LabelFrame(self.controls, text="Action Classification Controls")
        action_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Detection rate slider
        rate_frame = ttk.Frame(action_control_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Detection Rate (s):").pack(side=tk.LEFT, padx=5)
        self.detection_rate_var = tk.DoubleVar(value=3.0)  # Default 3.0 seconds
        rate_scale = ttk.Scale(rate_frame, from_=0.5, to=10.0, 
                               variable=self.detection_rate_var, 
                               command=self.update_detection_rate)
        rate_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.detection_rate_label = ttk.Label(rate_frame, text="3.0")
        self.detection_rate_label.pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold slider
        threshold_frame = ttk.Frame(action_control_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text="Confidence Threshold (%):").pack(side=tk.LEFT, padx=5)
        self.confidence_threshold_var = tk.DoubleVar(value=20.0)  # Default 20%
        threshold_scale = ttk.Scale(threshold_frame, from_=5.0, to=95.0, 
                                   variable=self.confidence_threshold_var, 
                                   command=self.update_confidence_threshold)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.threshold_label = ttk.Label(threshold_frame, text="20.0")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Detection control buttons
        buttons_frame = ttk.Frame(action_control_frame)
        buttons_frame.pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Faster Detection", 
                  style='Modern.TButton', 
                  command=self.increase_detection_rate).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(buttons_frame, text="Slower Detection", 
                  style='Modern.TButton', 
                  command=self.decrease_detection_rate).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add AI Coaching controls if available
        if self.coach_available:
            ai_coaching_frame = ttk.LabelFrame(self.controls, text="AI Coaching")
            ai_coaching_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Exercise selection
            exercise_frame = ttk.Frame(ai_coaching_frame)
            exercise_frame.pack(fill=tk.X, pady=2)
            ttk.Label(exercise_frame, text="Exercise:").pack(side=tk.LEFT, padx=5)
            
            self.exercise_var = tk.StringVar(value="Select Exercise")
            exercise_combobox = ttk.Combobox(exercise_frame, 
                                           textvariable=self.exercise_var,
                                           values=SUPPORTED_EXERCISES)
            exercise_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Start/End Exercise buttons
            button_frame = ttk.Frame(ai_coaching_frame)
            button_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(button_frame, text="Start Exercise", 
                     style='Modern.TButton', 
                     command=self.start_exercise).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            ttk.Button(button_frame, text="End Exercise", 
                     style='Modern.TButton', 
                     command=self.end_exercise).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Feedback toggle
            feedback_frame = ttk.Frame(ai_coaching_frame)
            feedback_frame.pack(fill=tk.X, pady=2)
            
            self.feedback_visible_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(feedback_frame, text="Show Feedback", 
                         variable=self.feedback_visible_var,
                         command=self.toggle_feedback_display).pack(side=tk.LEFT, padx=5)
            
            # Feedback frequency slider
            ttk.Label(feedback_frame, text="Feedback Interval:").pack(side=tk.LEFT, padx=5)
            self.feedback_interval_var = tk.DoubleVar(value=3.0)
            feedback_scale = ttk.Scale(feedback_frame, from_=1.0, to=10.0,
                                    variable=self.feedback_interval_var,
                                    command=self.update_feedback_interval)
            feedback_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.feedback_interval_label = ttk.Label(feedback_frame, text="3.0s")
            self.feedback_interval_label.pack(side=tk.LEFT, padx=5)
        
        # Add analysis data variables
        self.analysis_window = None
        self.analysis_type = None
        self.last_analysis_data = {
            'cosine_similarity': [],
            'dtw': [],
            'fourier': []
        }
    
    def on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """When the canvas is resized, resize the inner frame to match"""
        width = event.width
        self.control_canvas.itemconfig(self.control_canvas.find_withtag("all")[0], width=width)

    def connect_sensor(self):
        """Connect to the Phyphox sensor"""
        # Check if requests module is available
        if not REQUESTS_AVAILABLE:
            print("Error: requests package is not installed. Sensor functionality is disabled.")
            print("Install it with: pip install requests")
            self.connection_status.delete("all")
            self.connection_status.create_oval(0, 0, 20, 20, fill='red')
            return
            
        try:
            ip = self.ip_entry.get()
            port = self.port_entry.get()
            
            if not ip or not port:
                print("Error: IP and port are required")
                return
                
            # Store sensor details
            self.sensor_ip = ip
            self.sensor_port = port
            self.last_known_ip = ip
            self.last_known_port = port
            
            # Reset retry count
            self.connection_retry_count = 0
            
            # Update status to connecting
            self.connection_status.delete("all")
            self.connection_status.create_oval(0, 0, 20, 20, fill='orange')
            
            # Test connection
            try:
                response = requests.get(f"http://{ip}:{port}/get", timeout=0.5)
                if response.status_code == 200:
                    print(f"Successfully connected to sensor at {ip}:{port}")
                    self.connection_status.delete("all")
                    self.connection_status.create_oval(0, 0, 20, 20, fill='green')
                    self.sensor_connected = True
                    print("Sensor measurements started")
                else:
                    print(f"Failed to connect to sensor: HTTP {response.status_code}")
                    self.connection_status.delete("all")
                    self.connection_status.create_oval(0, 0, 20, 20, fill='red')
                    self.sensor_connected = False
            except Exception as e:
                print(f"Error connecting to sensor: {e}")
                self.connection_status.delete("all")
                self.connection_status.create_oval(0, 0, 20, 20, fill='red')
                self.sensor_connected = False
                
        except Exception as e:
            print(f"Error in connect_sensor: {e}")
            self.connection_status.delete("all")
            self.connection_status.create_oval(0, 0, 20, 20, fill='red')
    
    def reconnect_sensor(self):
        """Attempt to reconnect to the last known sensor"""
        try:
            if hasattr(self, 'last_known_ip') and hasattr(self, 'last_known_port'):
                if self.last_known_ip and self.last_known_port:
                    print(f"Attempting to reconnect to sensor at {self.last_known_ip}:{self.last_known_port}")
                    
                    # Update the IP and port entries for user feedback
                    self.ip_entry.delete(0, tk.END)
                    self.ip_entry.insert(0, self.last_known_ip)
                    
                    self.port_entry.delete(0, tk.END)
                    self.port_entry.insert(0, str(self.last_known_port))
                    
                    # Reset connection status
                    self.connection_retry_count = 0
                    self.sensor_ip = self.last_known_ip
                    self.sensor_port = self.last_known_port
                    
                    # Update UI status
                    self.connection_status.delete("all")
                    self.connection_status.create_oval(0, 0, 20, 20, fill='orange')
                    
                    # Force an immediate connection check
                    self.last_sensor_check = 0
                    self.update_sensor_data()
                    
                    return True
                else:
                    print("No previous connection details available")
            else:
                print("No previous connection details available")
                
            return False
        except Exception as e:
            print(f"Error reconnecting to sensor: {e}")
            return False
    
    def update_sensor_data(self):
        """Update sensor data from IMU sensors"""
        # This would be implemented to pull data from connected sensors
        # For demo purposes, we're generating simulated data
        pass

    def on_mouse_down(self, event):
        """Handle mouse down event for 3D view"""
        try:
            # Store initial mouse position
            self.is_dragging = True
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            
            # Pass event to pose_3d
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                # Create event dict that matches what handle_mouse_event expects
                mouse_event = {
                    'x': event.x,
                    'y': event.y,
                    'button': 1,  # Left mouse button
                    'state': event.state
                }
                self.pose_estimator.pose_3d.is_dragging = True
                self.pose_estimator.pose_3d.last_x = event.x
                self.pose_estimator.pose_3d.last_y = event.y
        except Exception as e:
            print(f"Error in on_mouse_down: {e}")

    def on_mouse_up(self, event):
        """Handle mouse up event for 3D view"""
        try:
            # End drag operation
            self.is_dragging = False
            
            # Pass event to pose_3d
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                # Create event dict that matches what handle_mouse_event expects
                mouse_event = {
                    'x': event.x,
                    'y': event.y,
                    'button': 1,  # Left mouse button
                    'state': event.state
                }
                self.pose_estimator.pose_3d.is_dragging = False
        except Exception as e:
            print(f"Error in on_mouse_up: {e}")

    def on_mouse_drag(self, event):
        """Handle mouse drag event for 3D view"""
        try:
            if not self.is_dragging:
                return
                
            # Calculate movement
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            # Update manually instead of using handle_mouse_event
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                pose_3d = self.pose_estimator.pose_3d
                
                # Check for right mouse button (state=1024) or middle button (state=2048)
                if event.state & 0x0400 or event.state & 0x0800:  # Right or middle mouse button
                    # Pan the camera
                    pose_3d.camera_x += dx * 0.01
                    pose_3d.camera_y -= dy * 0.01
                # Check for left mouse button (state=256)
                elif event.state & 0x0100:  
                    # Rotate the view
                    pose_3d.rotation_y += dx * 0.5
                    pose_3d.rotation_x += dy * 0.5
                # Fallback for other platforms or configurations
                else:
                    # Try to detect mouse button from event attributes
                    if hasattr(event, 'num') and event.num == 3:  # Right click
                        # Pan the camera
                        pose_3d.camera_x += dx * 0.01
                        pose_3d.camera_y -= dy * 0.01
                    else:  # Default to rotation with left click
                        pose_3d.rotation_y += dx * 0.5
                        pose_3d.rotation_x += dy * 0.5
            
            # Update last position
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
        except Exception as e:
            print(f"Error in on_mouse_drag: {e}")
            print(f"Event state: {event.state}, Event type: {type(event)}")

    def on_mouse_wheel(self, event):
        """Handle mouse wheel event for 3D view"""
        try:
            # Pass event to pose_3d
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                # Handle zoom using delta
                # On Windows, event.delta is a multiple of 120
                zoom_direction = 1 if event.delta > 0 else -1
                
                # Adjust camera distance - move closer or further
                self.pose_estimator.pose_3d.camera_distance -= zoom_direction * 0.3
                
                # Clamp to reasonable values
                self.pose_estimator.pose_3d.camera_distance = max(2.0, min(10.0, self.pose_estimator.pose_3d.camera_distance))
        except Exception as e:
            print(f"Error in on_mouse_wheel: {e}")

    def switch_camera(self):
        """Switch between available cameras"""
        try:
            if self.cap is not None:
                # Store current camera index
                if not hasattr(self, 'current_camera'):
                    self.current_camera = 0
                
                # Release current camera
                self.cap.release()
                
                # Try next camera index
                next_camera = (self.current_camera + 1) % 4  # Try indices 0-3
                print(f"Trying camera {next_camera}...")
                
                # Try to open the next camera
                self.cap = cv2.VideoCapture(next_camera)
                
                if self.cap.isOpened():
                    print(f"Successfully switched to camera {next_camera}")
                    self.current_camera = next_camera
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                else:
                    print(f"Failed to open camera {next_camera}")
                    # Try to find any working camera
                    for i in range(4):
                        if i != next_camera:  # Don't try the same camera again
                            print(f"Trying camera {i}...")
                            self.cap = cv2.VideoCapture(i)
                            if self.cap.isOpened():
                                print(f"Successfully switched to camera {i}")
                                self.current_camera = i
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                break
                    
                    if not self.cap.isOpened():
                        print("No working camera found")
                        # Try to reopen the original camera
                        self.cap = cv2.VideoCapture(self.current_camera)
                        if not self.cap.isOpened():
                            print("Failed to reopen original camera")
                            tk.messagebox.showerror("Camera Error", "Could not find any working camera")
        
        except Exception as e:
            print(f"Error switching camera: {e}")
            # Try to initialize any camera as fallback
            for i in range(4):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Fallback: Switched to camera {i}")
                    self.current_camera = i
                    break

    def toggle_background(self):
        """Toggle between light and dark background for 3D visualization"""
        try:
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                self.pose_estimator.pose_3d.is_dark_mode = not self.pose_estimator.pose_3d.is_dark_mode
                print(f"Background mode: {'Dark' if self.pose_estimator.pose_3d.is_dark_mode else 'Light'}")
        except Exception as e:
            print(f"Error toggling background: {e}")

    def reset_view(self):
        """Reset the 3D visualization to the default view"""
        try:
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                # Reset camera settings
                self.pose_estimator.pose_3d.camera_distance = 5.0
                self.pose_estimator.pose_3d.camera_x = 0.0
                self.pose_estimator.pose_3d.camera_y = 0.0
                self.pose_estimator.pose_3d.rotation_x = 0.0
                self.pose_estimator.pose_3d.rotation_y = 0.0
                print("View reset to default")
        except Exception as e:
            print(f"Error resetting view: {e}")

    def update_sensor_visualization(self):
        """Update the sensor visualization based on selected sensor type"""
        # This function would be implemented to update the visualization based on the selected sensor type
        pass

    def update_sensor_data(self):
        """Update sensor data from Phyphox"""
        # Check if requests module is available
        if not REQUESTS_AVAILABLE:
            # No access to requests module, can't update
            return
            
        if not self.sensor_ip:
            # No sensor connected - attempt auto reconnect if enabled
            if self.auto_reconnect and hasattr(self, 'last_known_ip') and hasattr(self, 'last_known_port'):
                current_time = time.time()
                # Only attempt reconnection every 5 seconds to avoid flooding
                if not hasattr(self, 'last_reconnect_attempt') or current_time - self.last_reconnect_attempt > 5.0:
                    self.last_reconnect_attempt = current_time
                    print(f"Attempting to reconnect to last known sensor at {self.last_known_ip}:{self.last_known_port}...")
                    # Quietly try to reconnect in the background
                    self.sensor_ip = self.last_known_ip
                    self.sensor_port = self.last_known_port
            return
        
        # Limit how often we check the sensor connection
        current_time = time.time()
        if current_time - self.last_sensor_check < self.sensor_check_interval:
            # Skip this check if we checked recently
            return
            
        self.last_sensor_check = current_time
        
        # Get current timeout value from UI
        timeout = self.timeout_var.get() if hasattr(self, 'timeout_var') else 0.5
            
        try:
            # Optimize by getting both gyro and accel data in a single request
            data_url = f"http://{self.sensor_ip}:{self.sensor_port}/get?gyrX&gyrY&gyrZ&gyr_time&accX&accY&accZ&acc_time"
            try:
                response = requests.get(data_url, timeout=timeout)  # Use timeout from UI
                
                if response.status_code == 200:
                    # Reset retry count on successful data retrieval
                    self.connection_retry_count = 0
                    
                    # Update connection status to green
                    self.connection_status.delete("all")
                    self.connection_status.create_oval(0, 0, 20, 20, fill='green')
                    
                    # Process all data
                    sensor_data = response.json()
                    if 'buffer' in sensor_data:
                        data = sensor_data['buffer']
                        
                        # Process gyro data
                        if all(key in data for key in ['gyrX', 'gyrY', 'gyrZ', 'gyr_time']):
                            gyro_buffer = {
                                'gyrX': data['gyrX'],
                                'gyrY': data['gyrY'],
                                'gyrZ': data['gyrZ'],
                                'gyr_time': data['gyr_time']
                            }
                            self.process_gyro_data(gyro_buffer)
                    
                        # Process accel data
                        if all(key in data for key in ['accX', 'accY', 'accZ', 'acc_time']):
                            accel_buffer = {
                                'accX': data['accX'],
                                'accY': data['accY'],
                                'accZ': data['accZ'],
                                'acc_time': data['acc_time']
                            }
                            self.process_accel_data(accel_buffer)
                    
                    # Update analysis and visualization
                    self.update_analysis()
                    self.update_sensor_visualization()
                else:
                    self.connection_retry_count += 1
                    if self.connection_retry_count >= self.max_retries:
                        print(f"Sensor connection lost after {self.max_retries} failed attempts. Status code: {response.status_code}")
                        self.connection_status.delete("all")
                        self.connection_status.create_oval(0, 0, 20, 20, fill='red')
                        # Save last known good connection for auto-reconnect
                        self.last_known_ip = self.sensor_ip
                        self.last_known_port = self.sensor_port
                        self.sensor_ip = None  # Reset connection, but keep the last known values
                        self.connection_retry_count = 0
                    else:
                        print(f"Sensor connection issue, retry {self.connection_retry_count}/{self.max_retries}")
                        self.connection_status.delete("all")
                        self.connection_status.create_oval(0, 0, 20, 20, fill='orange')
            except requests.exceptions.RequestException as e:
                self.connection_retry_count += 1
                if self.connection_retry_count >= self.max_retries:
                    print(f"Sensor connection lost after {self.max_retries} failed attempts: {e}")
                    self.connection_status.delete("all")
                    self.connection_status.create_oval(0, 0, 20, 20, fill='red')
                    # Save last known good connection for auto-reconnect
                    self.last_known_ip = self.sensor_ip
                    self.last_known_port = self.sensor_port
                    self.sensor_ip = None  # Reset connection, but keep the last known values
                    self.connection_retry_count = 0
                else:
                    print(f"Sensor connection issue, retry {self.connection_retry_count}/{self.max_retries}")
                    self.connection_status.delete("all")
                    self.connection_status.create_oval(0, 0, 20, 20, fill='orange')
                    
        except Exception as e:
            self.connection_retry_count += 1
            if self.connection_retry_count >= self.max_retries:
                print(f"Error updating sensor data: {e}")
                # Update connection status to red on error
                self.connection_status.delete("all")
                self.connection_status.create_oval(0, 0, 20, 20, fill='red')
                # Save last known good connection for auto-reconnect
                if self.sensor_ip:
                    self.last_known_ip = self.sensor_ip
                    self.last_known_port = self.sensor_port
                self.sensor_ip = None  # Reset connection, but keep the last known values
                self.connection_retry_count = 0
            else:
                print(f"Sensor data issue, retry {self.connection_retry_count}/{self.max_retries}")
                self.connection_status.delete("all")
                self.connection_status.create_oval(0, 0, 20, 20, fill='orange')

    def process_gyro_data(self, buffer):
        """Process gyroscope data using vectorized operations for better performance"""
        try:
            if not hasattr(self, 'sensor_data'):
                self.sensor_data = {
                    'gyro': {'x': [], 'y': [], 'z': [], 'magnitude': [], 'time': []},
                    'accel': {'x': [], 'y': [], 'z': [], 'magnitude': [], 'time': []},
                    'max_gyro': 0,
                    'avg_gyro': 0,
                    'max_accel': 0,
                    'avg_accel': 0
                }
                
            # Get data from buffer, filtering out None values
            x_data = [x for x in buffer['gyrX']['buffer'] if x is not None]
            y_data = [y for y in buffer['gyrY']['buffer'] if y is not None]
            z_data = [z for z in buffer['gyrZ']['buffer'] if z is not None]
            time_data = [t for t in buffer['gyr_time']['buffer'] if t is not None]
            
            # Ensure all arrays have the same length (using the shortest)
            min_length = min(len(x_data), len(y_data), len(z_data), len(time_data))
            
            if min_length == 0:
                return
                
            # Take only valid data points
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]
            z_data = z_data[:min_length]
            time_data = time_data[:min_length]
            
            # Add data to sensor_data in a batch
            self.sensor_data['gyro']['x'].extend(x_data)
            self.sensor_data['gyro']['y'].extend(y_data)
            self.sensor_data['gyro']['z'].extend(z_data)
            self.sensor_data['gyro']['time'].extend(time_data)
            
            # Calculate magnitude using numpy vectorization
            if min_length > 0:
                x_array = np.array(x_data)
                y_array = np.array(y_data)
                z_array = np.array(z_data)
                
                # Vectorized magnitude calculation
                magnitudes = np.sqrt(x_array**2 + y_array**2 + z_array**2)
                self.sensor_data['gyro']['magnitude'].extend(magnitudes)
            
            # Keep only the most recent data points (for better performance)
            max_points = 100
            for key in self.sensor_data['gyro']:
                self.sensor_data['gyro'][key] = self.sensor_data['gyro'][key][-max_points:]
                
            # Update metrics
            if self.sensor_data['gyro']['magnitude']:
                self.sensor_data['max_gyro'] = max([abs(x) for x in self.sensor_data['gyro']['magnitude']])
                self.sensor_data['avg_gyro'] = sum([abs(x) for x in self.sensor_data['gyro']['magnitude']]) / len(self.sensor_data['gyro']['magnitude'])
                
        except Exception as e:
            print(f"Error processing gyro data: {e}")

    def process_accel_data(self, buffer):
        """Process acceleration data using vectorized operations for better performance"""
        try:
            if not hasattr(self, 'sensor_data'):
                self.sensor_data = {
                    'gyro': {'x': [], 'y': [], 'z': [], 'magnitude': [], 'time': []},
                    'accel': {'x': [], 'y': [], 'z': [], 'magnitude': [], 'time': []},
                    'max_gyro': 0,
                    'avg_gyro': 0,
                    'max_accel': 0,
                    'avg_accel': 0
                }
                
            # Get data from buffer, filtering out None values
            x_data = [x for x in buffer['accX']['buffer'] if x is not None]
            y_data = [y for y in buffer['accY']['buffer'] if y is not None]
            z_data = [z for z in buffer['accZ']['buffer'] if z is not None]
            time_data = [t for t in buffer['acc_time']['buffer'] if t is not None]
            
            # Ensure all arrays have the same length (using the shortest)
            min_length = min(len(x_data), len(y_data), len(z_data), len(time_data))
            
            if min_length == 0:
                return
                
            # Take only valid data points
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]
            z_data = z_data[:min_length]
            time_data = time_data[:min_length]
            
            # Add data to sensor_data in a batch
            self.sensor_data['accel']['x'].extend(x_data)
            self.sensor_data['accel']['y'].extend(y_data)
            self.sensor_data['accel']['z'].extend(z_data)
            self.sensor_data['accel']['time'].extend(time_data)
            
            # Calculate magnitude using numpy vectorization
            if min_length > 0:
                x_array = np.array(x_data)
                y_array = np.array(y_data)
                z_array = np.array(z_data)
                
                # Vectorized magnitude calculation
                magnitudes = np.sqrt(x_array**2 + y_array**2 + z_array**2)
                self.sensor_data['accel']['magnitude'].extend(magnitudes)
            
            # Keep only the most recent data points (for better performance)
            max_points = 100
            for key in self.sensor_data['accel']:
                self.sensor_data['accel'][key] = self.sensor_data['accel'][key][-max_points:]
                
            # Update metrics
            if self.sensor_data['accel']['magnitude']:
                self.sensor_data['max_accel'] = max([abs(x) for x in self.sensor_data['accel']['magnitude']])
                self.sensor_data['avg_accel'] = sum([abs(x) for x in self.sensor_data['accel']['magnitude']]) / len(self.sensor_data['accel']['magnitude'])
                
        except Exception as e:
            print(f"Error processing accel data: {e}")

    def update_analysis(self):
        """Update the analysis display"""
        try:
            # Get the current sensor type
            sensor_type = self.sensor_type.get()
            data = self.sensor_data[sensor_type]
            
            if not data['time'] or not all(len(data[axis]) > 0 for axis in ['x', 'y', 'z']):
                return
                
            # Filter out None values and ensure arrays are the same length
            valid_indices = [i for i in range(len(data['time'])) 
                           if all(i < len(data[axis]) and data[axis][i] is not None for axis in ['x', 'y', 'z', 'time'])]
            
            if not valid_indices:
                return
                
            # Create clean arrays with only valid data
            clean_data = {
                'x': [data['x'][i] for i in valid_indices],
                'y': [data['y'][i] for i in valid_indices],
                'z': [data['z'][i] for i in valid_indices],
                'time': [data['time'][i] for i in valid_indices]
            }
            
            # Calculate statistics using actual sensor values
            if sensor_type == "gyro":
                # Calculate maximum and average for each axis
                max_x = max(abs(x) for x in clean_data['x'])
                max_y = max(abs(y) for y in clean_data['y'])
                max_z = max(abs(z) for z in clean_data['z'])
                
                # Calculate overall maximum and average
                self.sensor_data['max_gyro'] = max(max_x, max_y, max_z)
                self.sensor_data['avg_gyro'] = sum(np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(clean_data['x'], clean_data['y'], clean_data['z'])) / len(clean_data['x'])
                
            elif sensor_type == "accel":
                # Calculate maximum and average for each axis
                max_x = max(abs(x) for x in clean_data['x'])
                max_y = max(abs(y) for y in clean_data['y'])
                max_z = max(abs(z) for z in clean_data['z'])
                
                # Calculate overall maximum and average
                self.sensor_data['max_accel'] = max(max_x, max_y, max_z)
                self.sensor_data['avg_accel'] = sum(np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(clean_data['x'], clean_data['y'], clean_data['z'])) / len(clean_data['x'])
                
        except Exception as e:
            print(f"Error updating sensor analysis: {e}")

    def update_analysis_overlay(self):
        """Update the analysis overlay display"""
        try:
            # Draw directly on the colored_frame canvas
            canvas = self.colored_frame
            
            # Define position and size for the overlay - top right corner
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            overlay_width = 280
            overlay_height = 200
            overlay_x = canvas_width - overlay_width - 10
            overlay_y = 10
            
            # Clear previous content
            canvas.delete("analysis_overlay")
            
            # Draw background
            canvas.create_rectangle(
                overlay_x, overlay_y, 
                overlay_x + overlay_width, overlay_y + overlay_height, 
                fill='black', stipple='gray50', outline='white', width=2,
                tags="analysis_overlay"
            )
            
            # Draw title
            canvas.create_text(
                overlay_x + overlay_width//2, overlay_y + 15, 
                text="Motion Analysis", 
                fill='white', font=('Helvetica', 10, 'bold'),
                tags="analysis_overlay"
            )
            
            # Get current sensor type and data
            sensor_type = self.sensor_type.get()
            
            if sensor_type not in self.sensor_data or not self.sensor_data[sensor_type]['time']:
                return
                
            data = self.sensor_data[sensor_type]
            
            # Draw statistics based on sensor type with smaller font
            y_pos = overlay_y + 35
            if sensor_type == "gyro":
                stats = [
                    f"Max X: {max(abs(x) for x in data['x'] if x is not None) if data['x'] else 0:.1f} rad/s",
                    f"Max Y: {max(abs(y) for y in data['y'] if y is not None) if data['y'] else 0:.1f} rad/s",
                    f"Max Z: {max(abs(z) for z in data['z'] if z is not None) if data['z'] else 0:.1f} rad/s",
                    f"Max Rotation: {self.sensor_data['max_gyro']:.1f} rad/s",
                    f"Avg Rotation: {self.sensor_data['avg_gyro']:.1f} rad/s"
                ]
            else:  # accelerometer
                stats = [
                    f"Max X: {max(abs(x) for x in data['x'] if x is not None) if data['x'] else 0:.1f} m/s²",
                    f"Max Y: {max(abs(y) for y in data['y'] if y is not None) if data['y'] else 0:.1f} m/s²",
                    f"Max Z: {max(abs(z) for z in data['z'] if z is not None) if data['z'] else 0:.1f} m/s²",
                    f"Max Accel: {self.sensor_data['max_accel']:.2f} m/s²",
                    f"Avg Accel: {self.sensor_data['avg_accel']:.2f} m/s²"
                ]
            
            for stat in stats:
                canvas.create_text(
                    overlay_x + overlay_width//2, y_pos, 
                    text=stat, fill='white', font=('Helvetica', 8),
                    tags="analysis_overlay"
                )
                y_pos += 20
            
            # Draw real-time indicators
            if data['x']:
                # Current X value
                current_x = data['x'][-1] if data['x'] else 0
                max_x = max(abs(x) for x in data['x'] if x is not None) if data['x'] else 1
                x_bar_width = min(200, (abs(current_x) / max(1, max_x)) * 200)
                
                bar_y = overlay_y + 150
                canvas.create_rectangle(
                    overlay_x + 50, bar_y, 
                    overlay_x + 50 + x_bar_width, bar_y + 15, 
                    fill='red', tags="analysis_overlay"
                )
                canvas.create_text(
                    overlay_x + 250, bar_y + 7, 
                    text=f"X: {current_x:.1f}", 
                    fill='white', font=('Helvetica', 8),
                    tags="analysis_overlay"
                )
                
                # Current Y value
                current_y = data['y'][-1] if data['y'] else 0
                max_y = max(abs(y) for y in data['y'] if y is not None) if data['y'] else 1
                y_bar_width = min(200, (abs(current_y) / max(1, max_y)) * 200)
                
                bar_y = overlay_y + 170
                canvas.create_rectangle(
                    overlay_x + 50, bar_y, 
                    overlay_x + 50 + y_bar_width, bar_y + 15, 
                    fill='green', tags="analysis_overlay"
                )
                canvas.create_text(
                    overlay_x + 250, bar_y + 7, 
                    text=f"Y: {current_y:.1f}", 
                    fill='white', font=('Helvetica', 8),
                    tags="analysis_overlay"
                )
                
        except Exception as e:
            print(f"Error updating analysis overlay: {e}")

    def update_motion_analysis_overlay(self):
        """Update motion analysis overlay with latest sensor data and visualizations"""
        try:
            # Clear previous overlay
            if hasattr(self, 'colored_frame'):
                self.colored_frame.delete("motion_analysis")
                
                # Get canvas dimensions for positioning
                canvas_width = self.colored_frame.winfo_width()
                canvas_height = self.colored_frame.winfo_height()
                
                # Define dimensions and position for the overlay
                overlay_width = 300  # Increased width to accommodate graphs
                overlay_height = 290  # Reduced height to prevent overlap
                overlay_x = 10  # Position on left side
                overlay_y = 10  # Position at top
                
                # Create semi-transparent background
                bg_color = "black"
                self.colored_frame.create_rectangle(
                    overlay_x, overlay_y,
                    overlay_x + overlay_width, overlay_y + overlay_height,
                    fill=bg_color, stipple='gray50', outline='white',
                    tags="motion_analysis"
                )
                
                # Add title
                self.colored_frame.create_text(
                    overlay_x + overlay_width//2, overlay_y + 15,
                    text="Motion Analysis",
                    fill='white', font=('Arial', 12, 'bold'),
                    tags="motion_analysis"
                )
                
                # Get the current sensor type (gyro or accel)
                sensor_type = self.sensor_type.get() if hasattr(self, 'sensor_type') else "gyro"
                
                # Add sensor type indicator
                self.colored_frame.create_text(
                    overlay_x + overlay_width//2, overlay_y + 35,
                    text=f"Sensor: {sensor_type.capitalize()}",
                    fill='yellow', font=('Arial', 9, 'bold'),
                    tags="motion_analysis"
                )
                
                # Show sensor specific metrics
                if hasattr(self, 'sensor_data') and self.sensor_data[sensor_type]['x']:
                    # Current values section - add current x, y, z values
                    self.colored_frame.create_text(
                        overlay_x + 10, overlay_y + 55,
                        text="Current Values:",
                        fill='white', font=('Arial', 9, 'bold'),
                        anchor=tk.W,
                        tags="motion_analysis"
                    )
                    
                    # Get latest values
                    latest_x = self.sensor_data[sensor_type]['x'][-1] if self.sensor_data[sensor_type]['x'] else 0
                    latest_y = self.sensor_data[sensor_type]['y'][-1] if self.sensor_data[sensor_type]['y'] else 0
                    latest_z = self.sensor_data[sensor_type]['z'][-1] if self.sensor_data[sensor_type]['z'] else 0
                    
                    # Add color-coded x, y, z values
                    self.colored_frame.create_text(
                        overlay_x + 15, overlay_y + 75,
                        text=f"X: {latest_x:.2f}",
                        fill='red', font=('Arial', 9),
                        anchor=tk.W,
                        tags="motion_analysis"
                    )
                    
                    self.colored_frame.create_text(
                        overlay_x + 110, overlay_y + 75,
                        text=f"Y: {latest_y:.2f}",
                        fill='green', font=('Arial', 9),
                        anchor=tk.W,
                        tags="motion_analysis"
                    )
                    
                    self.colored_frame.create_text(
                        overlay_x + 205, overlay_y + 75,
                        text=f"Z: {latest_z:.2f}",
                        fill='blue', font=('Arial', 9),
                        anchor=tk.W,
                        tags="motion_analysis"
                    )
                    
                    # Show average and max for the current sensor type
                    if sensor_type == "gyro":
                        # Get rotation metrics
                        avg_rotation = self.sensor_data['avg_gyro']
                        max_rotation = self.sensor_data['max_gyro']
                        unit = "rad/s"
                        
                        # Update metric labels
                        self.colored_frame.create_text(
                            overlay_x + 10, overlay_y + 95,
                            text=f"Avg Rotation: {avg_rotation:.2f} {unit}",
                            fill='white', font=('Arial', 9),
                            anchor=tk.W,
                            tags="motion_analysis"
                        )
                        
                        self.colored_frame.create_text(
                            overlay_x + 10, overlay_y + 115,
                            text=f"Max Rotation: {max_rotation:.2f} {unit}",
                            fill='white', font=('Arial', 9),
                            anchor=tk.W,
                            tags="motion_analysis"
                        )
                    else:  # accel
                        # Get acceleration metrics
                        avg_accel = self.sensor_data['avg_accel']
                        max_accel = self.sensor_data['max_accel']
                        unit = "m/s²"
                        
                        # Update metric labels
                        self.colored_frame.create_text(
                            overlay_x + 10, overlay_y + 95,
                            text=f"Avg Accel: {avg_accel:.2f} {unit}",
                            fill='white', font=('Arial', 9),
                            anchor=tk.W,
                            tags="motion_analysis"
                        )
                        
                        self.colored_frame.create_text(
                            overlay_x + 10, overlay_y + 115,
                            text=f"Max Accel: {max_accel:.2f} {unit}",
                            fill='white', font=('Arial', 9),
                            anchor=tk.W,
                            tags="motion_analysis"
                        )
                    
                    # Add title for bar graphs
                    self.colored_frame.create_text(
                        overlay_x + overlay_width//2, overlay_y + 135,
                        text="Sensor Data Graphs",
                        fill='white', font=('Arial', 9, 'bold'),
                        tags="motion_analysis"
                    )
                    
                    # Calculate rate of change for each axis (if we have enough data points)
                    data_x = self.sensor_data[sensor_type]['x']
                    data_y = self.sensor_data[sensor_type]['y']
                    data_z = self.sensor_data[sensor_type]['z']
                    
                    # Calculate rate of change (derivative) for last few points
                    window = 10  # Use last 10 points for rate calculation
                    if len(data_x) >= window:
                        # Get rate of change by looking at difference between latest and previous values
                        roc_x = abs(data_x[-1] - data_x[-window]) if len(data_x) >= window else abs(data_x[-1])
                        roc_y = abs(data_y[-1] - data_y[-window]) if len(data_y) >= window else abs(data_y[-1])
                        roc_z = abs(data_z[-1] - data_z[-window]) if len(data_z) >= window else abs(data_z[-1])
                        
                        # Normalize for display (scale to 0-100 range)
                        max_roc = max(roc_x, roc_y, roc_z, 0.1)  # Avoid division by zero
                        bar_scale = 100 / max_roc
                        
                        # Cap at 100 to prevent overflow
                        bar_x = min(roc_x * bar_scale, 100)
                        bar_y = min(roc_y * bar_scale, 100)
                        bar_z = min(roc_z * bar_scale, 100)
                        
                        # Draw bar graphs for rate of change
                        # X-axis bar (red)
                        bar_height = 18
                        bar_spacing = 25
                        bar_width = 235
                        start_y = overlay_y + 155
                        
                        # X bar and frame
                        self.colored_frame.create_rectangle(
                            overlay_x + 40, start_y,
                            overlay_x + 40 + bar_width, start_y + bar_height,
                            fill="black", outline="white",
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_rectangle(
                            overlay_x + 40, start_y,
                            overlay_x + 40 + (bar_width * bar_x / 100), start_y + bar_height,
                            fill="red", outline="",
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_text(
                            overlay_x + 20, start_y + bar_height/2,
                            text="X:",
                            fill="red", font=('Arial', 9, 'bold'),
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_text(
                            overlay_x + 40 + bar_width + 10, start_y + bar_height/2,
                            text=f"{roc_x:.2f}",
                            fill="red", font=('Arial', 9),
                            tags="motion_analysis"
                        )
                        
                        # Y bar and frame
                        self.colored_frame.create_rectangle(
                            overlay_x + 40, start_y + bar_spacing,
                            overlay_x + 40 + bar_width, start_y + bar_spacing + bar_height,
                            fill="black", outline="white",
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_rectangle(
                            overlay_x + 40, start_y + bar_spacing,
                            overlay_x + 40 + (bar_width * bar_y / 100), start_y + bar_spacing + bar_height,
                            fill="green", outline="",
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_text(
                            overlay_x + 20, start_y + bar_spacing + bar_height/2,
                            text="Y:",
                            fill="green", font=('Arial', 9, 'bold'),
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_text(
                            overlay_x + 40 + bar_width + 10, start_y + bar_spacing + bar_height/2,
                            text=f"{roc_y:.2f}",
                            fill="green", font=('Arial', 9),
                            tags="motion_analysis"
                        )
                        
                        # Z bar and frame
                        self.colored_frame.create_rectangle(
                            overlay_x + 40, start_y + 2*bar_spacing,
                            overlay_x + 40 + bar_width, start_y + 2*bar_spacing + bar_height,
                            fill="black", outline="white",
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_rectangle(
                            overlay_x + 40, start_y + 2*bar_spacing,
                            overlay_x + 40 + (bar_width * bar_z / 100), start_y + 2*bar_spacing + bar_height,
                            fill="blue", outline="",
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_text(
                            overlay_x + 20, start_y + 2*bar_spacing + bar_height/2,
                            text="Z:",
                            fill="blue", font=('Arial', 9, 'bold'),
                            tags="motion_analysis"
                        )
                        self.colored_frame.create_text(
                            overlay_x + 40 + bar_width + 10, start_y + 2*bar_spacing + bar_height/2,
                            text=f"{roc_z:.2f}",
                            fill="blue", font=('Arial', 9),
                            tags="motion_analysis"
                        )
                        
                        # Add explanation of what the bars mean (with smaller font)
                        if sensor_type == "gyro":
                            explanation = "Bars show rate of rotation change"
                        else:
                            explanation = "Bars show rate of acceleration change"
                            
                        self.colored_frame.create_text(
                            overlay_x + overlay_width//2, start_y + 2*bar_spacing + 25,
                            text=explanation,
                            fill='white', font=('Arial', 8, 'italic'),
                            tags="motion_analysis"
                        )
                    else:
                        # Not enough data for rate of change calculation
                        self.colored_frame.create_text(
                            overlay_x + overlay_width//2, overlay_y + 180,
                            text="Waiting for more data...",
                            fill='gray', font=('Arial', 9),
                            tags="motion_analysis"
                        )
                    
                    # Add click indicator to show this syncs with sensor data selection
                    self.colored_frame.create_text(
                        overlay_x + overlay_width//2, overlay_y + 275,
                        text="Updates based on selected sensor type",
                        fill='gray', font=('Arial', 8, 'italic'),
                        tags="motion_analysis"
                    )
                else:
                    # No sensor data available
                    self.colored_frame.create_text(
                        overlay_x + overlay_width//2, overlay_y + 150,
                        text="Waiting for sensor data...",
                        fill='gray', font=('Arial', 10),
                        tags="motion_analysis"
                    )
        except Exception as e:
            print(f"Error updating motion analysis overlay: {e}")
            import traceback
            traceback.print_exc()

    def update_reconnect_setting(self):
        """Update the auto reconnect setting"""
        self.auto_reconnect = self.auto_reconnect_var.get()
        print(f"Auto reconnect {'enabled' if self.auto_reconnect else 'disabled'}")
        
    def update_timeout(self, *args):
        """Update the sensor connection timeout"""
        timeout = self.timeout_var.get()
        self.timeout_label.config(text=f"{timeout:.1f}")
        print(f"Sensor connection timeout set to {timeout:.1f} seconds")
        
    def update_check_interval(self, *args):
        """Update the sensor check interval"""
        interval = self.interval_var.get()
        self.interval_label.config(text=f"{interval:.1f}")
        self.sensor_check_interval = interval
        print(f"Sensor check interval set to {interval:.1f} seconds")
        
    def update_max_retries(self):
        """Update the maximum retry count"""
        retries = self.max_retries_var.get()
        self.max_retries = retries
        print(f"Maximum connection retries set to {retries}")

    def toggle_sensor_settings(self):
        """Toggle the visibility of sensor settings"""
        if self.sensor_settings_visible:
            self.sensor_settings_frame.pack_forget()
            self.sensor_settings_visible = False
        else:
            self.sensor_settings_frame.pack(fill='x', padx=5, pady=5)
            self.sensor_settings_visible = True
            
    def update_data_points(self):
        """Update the number of data points to display"""
        try:
            points = self.data_points_var.get()
            # Store in a class variable for use in sensor data processing
            self.max_sensor_points = points
            print(f"Sensor data display points set to {points}")
        except Exception as e:
            print(f"Error updating data points: {e}")

    def show_exercise_dropdown(self):
        """Show the exercise selection dropdown when the exercise button is clicked"""
        if not hasattr(self.exercise_dropdown, 'winfo_ismapped') or not self.exercise_dropdown.winfo_ismapped():
            self.exercise_dropdown.pack(side='left', padx=5, pady=5)
        else:
            self.exercise_dropdown.pack_forget()
        
    def on_exercise_selected(self, event):
        """Handle exercise selection from dropdown"""
        selected = self.exercise_var.get()
        if selected and self.ai_coach:
            # Configure the AI coach with the selected exercise
            self.ai_coach.set_exercise(selected)
            # Show the start button
            self.exercise_control_button.pack(side='left', padx=5, pady=5)
            self.timer_label.pack(side='left', padx=5, pady=5)
            # Show the feedback panel
            self.feedback_panel.place(x=10, y=60, width=280, height=150)
            self.feedback_text.delete(1.0, tk.END)
            self.feedback_text.insert(tk.END, f"Selected {selected}. Press Start to begin.")
            # Show metrics panel
            self.metrics_panel.place(x=10, y=220, width=250, height=200)
            
    def toggle_exercise(self):
        """Start or stop the selected exercise"""
        if not self.exercise_active:
            # Start exercise
            self.exercise_active = True
            self.exercise_timer_start = time.time()
            self.exercise_control_button.config(text="Stop Exercise", bg='red')
            
            # Start exercise session in AI coach
            if self.ai_coach:
                self.ai_coach.start_session()
                self.feedback_text.delete(1.0, tk.END)
                self.feedback_text.insert(tk.END, f"Exercise started. AI coach is active.\n\nPerform the exercise and receive feedback.")
                
            # Start timer update
            self.update_exercise_timer()
            
            # Clear metrics display
            self.metrics_canvas.delete("all")
        else:
            # Stop exercise
            self.exercise_active = False
            self.exercise_control_button.config(text="Start Exercise", bg='green')
            
            # End session and get final feedback
            if self.ai_coach:
                final_feedback = self.ai_coach.end_session()
                if final_feedback:
                    self.feedback_text.delete(1.0, tk.END)
                    self.feedback_text.insert(tk.END, f"Exercise completed.\n\n{final_feedback}")
    
    def update_exercise_timer(self):
        """Update the exercise timer display"""
        if self.exercise_active and self.exercise_timer_start:
            current_time = time.time()
            elapsed = current_time - self.exercise_timer_start
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
            
            # Schedule next update
            self.root.after(1000, self.update_exercise_timer)
    
    def update_exercise_metrics(self, landmarks):
        """Update exercise metrics based on pose landmarks"""
        if self.exercise_active and self.ai_coach and landmarks:
            # Convert landmarks to the format expected by the AI coach
            landmark_list = []
            for landmark in landmarks:
                landmark_list.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # Update AI coach with new pose data
            feedback = self.ai_coach.update_metrics(landmark_list)
            if feedback:
                # Display the feedback
                self.feedback_text.delete(1.0, tk.END)
                self.feedback_text.insert(tk.END, feedback)
            
            # Update metrics visualization
            self.draw_exercise_metrics()
    
    def draw_exercise_metrics(self):
        """Draw exercise metrics visualization on the metrics canvas"""
        if not self.ai_coach:
            return
            
        self.metrics_canvas.delete("all")
        metrics = self.ai_coach.get_metrics()
        if not metrics:
            return
            
        # Draw exercise-specific metrics
        y_pos = 20
        for metric_name, metric_value in metrics.items():
            # Draw metric name
            self.metrics_canvas.create_text(10, y_pos, text=f"{metric_name}:", 
                                          fill="white", anchor="w", font=('Helvetica', 10))
            
            # Draw metric value bar
            bar_width = min(200, int(metric_value * 200))  # Scale to max 200px
            self.metrics_canvas.create_rectangle(100, y_pos-8, 100+bar_width, y_pos+8, 
                                               fill="green" if metric_value > 0.7 else "yellow" if metric_value > 0.4 else "red",
                                               outline="")
            
            # Draw percentage text
            self.metrics_canvas.create_text(100+bar_width+10, y_pos, text=f"{int(metric_value*100)}%", 
                                          fill="white", anchor="w", font=('Helvetica', 9))
            
            y_pos += 30
            
        # Draw rep counter if available
        if 'reps' in metrics:
            self.metrics_canvas.create_text(10, y_pos+10, text=f"Repetitions: {metrics['reps']}", 
                                          fill="cyan", anchor="w", font=('Helvetica', 12, 'bold'))

    def update_frame(self):
        """Update the frame with new data from camera and process it."""
        ret, frame = self.vid.read()
        if ret:
            # Process frame with pose estimator
            frame, landmarks, landmark_world = self.pose_estimator.process_frame(frame)
            
            # Update exercise metrics if active
            if self.exercise_active and landmarks and self.ai_coach:
                self.update_exercise_metrics(landmarks)
            
            # Use action classifier to detect activities if available
            if self.action_classifier:
                try:
                    # Prepare input for action classifier
                    action, confidence = self.action_classifier.predict_action(landmark_world)
                    
                    if action and confidence > 0.4:  # Only show if confidence is reasonable
                        # Display detected action on frame
                        action_text = f"{action}: {confidence:.1%}"
                        cv2.putText(frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error in action classification: {e}")
            
            # Convert and display the frame
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # Update the sensor data if available
            self.update_sensor_data()
        
        # Call this function again after delay
        self.root.after(self.delay, self.update_frame)

    def on_middle_mouse_down(self, event):
        """Handle middle mouse button press for panning"""
        try:
            self.is_dragging = True
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
        except Exception as e:
            print(f"Error in middle mouse down: {e}")

    def on_right_mouse_down(self, event):
        """Handle right mouse button press for panning"""
        try:
            self.is_dragging = True
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
        except Exception as e:
            print(f"Error in right mouse down: {e}")

    def on_middle_mouse_drag(self, event):
        """Handle middle mouse button dragging for panning"""
        try:
            if not self.is_dragging:
                return
                
            # Calculate movement
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            # Update pan position
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                self.pose_estimator.pose_3d.camera_x += dx * 0.01
                self.pose_estimator.pose_3d.camera_y -= dy * 0.01
            
            # Update last position
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
        except Exception as e:
            print(f"Error in middle mouse drag: {e}")

    def on_right_mouse_drag(self, event):
        """Handle right mouse button dragging for panning"""
        try:
            if not self.is_dragging:
                return
                
            # Calculate movement
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            # Update pan position
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                self.pose_estimator.pose_3d.camera_x += dx * 0.01
                self.pose_estimator.pose_3d.camera_y -= dy * 0.01
            
            # Update last position
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
        except Exception as e:
            print(f"Error in right mouse drag: {e}")

    def on_key_press(self, event):
        """Handle keyboard navigation for 3D view"""
        try:
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                pose_3d = self.pose_estimator.pose_3d
                
                # Arrow keys for panning
                if event.keysym == 'Up':
                    pose_3d.camera_y += 0.1
                elif event.keysym == 'Down':
                    pose_3d.camera_y -= 0.1
                elif event.keysym == 'Left':
                    pose_3d.camera_x -= 0.1
                elif event.keysym == 'Right':
                    pose_3d.camera_x += 0.1
                # WASD for rotation
                elif event.keysym.lower() == 'w':
                    pose_3d.rotation_x -= 5
                elif event.keysym.lower() == 's':
                    pose_3d.rotation_x += 5
                elif event.keysym.lower() == 'a':
                    pose_3d.rotation_y -= 5
                elif event.keysym.lower() == 'd':
                    pose_3d.rotation_y += 5
                # + and - for zoom
                elif event.keysym in ['plus', 'equal']:
                    pose_3d.camera_distance = max(2.0, pose_3d.camera_distance - 0.3)
                elif event.keysym in ['minus', 'underscore']:
                    pose_3d.camera_distance = min(10.0, pose_3d.camera_distance + 0.3)
                # Space to reset view
                elif event.keysym == 'space':
                    self.reset_view()
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def update_detection_rate(self, *args):
        """Update action classifier detection rate"""
        try:
            rate = self.detection_rate_var.get()
            self.detection_rate_label.config(text=f"{rate:.1f}")
            
            # Update the prediction interval in the action classifier
            if hasattr(self, 'action_classifier') and self.action_classifier:
                self.action_classifier.prediction_interval = rate
                print(f"Action detection rate set to {rate:.1f} seconds")
        except Exception as e:
            print(f"Error updating detection rate: {e}")

    def increase_detection_rate(self):
        """Increase action detection rate (make faster)"""
        try:
            current = self.detection_rate_var.get()
            new_rate = max(0.5, current - 0.5)  # Decrease interval (faster detection)
            self.detection_rate_var.set(new_rate)
            self.update_detection_rate()
        except Exception as e:
            print(f"Error increasing detection rate: {e}")

    def decrease_detection_rate(self):
        """Decrease action detection rate (make slower)"""
        try:
            current = self.detection_rate_var.get()
            new_rate = min(10.0, current + 0.5)  # Increase interval (slower detection)
            self.detection_rate_var.set(new_rate)
            self.update_detection_rate()
        except Exception as e:
            print(f"Error decreasing detection rate: {e}")

    def update_confidence_threshold(self, *args):
        """Update action classification confidence threshold"""
        try:
            threshold = self.confidence_threshold_var.get()
            self.threshold_label.config(text=f"{threshold:.1f}")
            print(f"Action confidence threshold set to {threshold:.1f}%")
        except Exception as e:
            print(f"Error updating confidence threshold: {e}")

    # Add AI coach methods
    def start_exercise(self):
        """Start a coaching session for the selected exercise"""
        if not self.coach_available:
            messagebox.showerror("Coach Unavailable", 
                               "AI Coach is not available. Make sure ai_feedback.py is properly installed.")
            return
            
        exercise = self.exercise_var.get()
        
        if exercise not in SUPPORTED_EXERCISES:
            messagebox.showwarning("Invalid Exercise", 
                                 "Please select a valid exercise from the dropdown.")
            return
        
        feedback = self.ai_coach.start_exercise_session(exercise)
        self.current_feedback = feedback
        self.exercise_active = True
        
        # Reset metrics
        self.exercise_metrics = {
            'rep_count': 0,
            'form_score': 0,
            'speed': 0,
            'range_of_motion': 0,
            'stability': 0
        }
        
        # Start timer
        self.timer_start = time.time()
        
        print(f"Started exercise session: {exercise}")
    
    def end_exercise(self):
        """End the current exercise session"""
        if not self.coach_available:
            messagebox.showerror("Coach Unavailable", 
                               "AI Coach is not available. Make sure ai_feedback.py is properly installed.")
            return
            
        if not self.exercise_active:
            messagebox.showinfo("No Active Session", "No exercise session is currently active.")
            return
        
        result = self.ai_coach.end_exercise_session()
        self.current_feedback = result.get('summary', 'Exercise session ended.')
        
        # Display tips in a messagebox
        tips = result.get('tips', [])
        if tips:
            messagebox.showinfo("Improvement Tips", 
                             "\n".join([f"• {tip}" for tip in tips]))
        
        self.exercise_active = False
        print("Ended exercise session")
    
    def toggle_feedback_display(self):
        """Toggle the visibility of the feedback display"""
        self.feedback_visible = self.feedback_visible_var.get()
        print(f"Feedback display: {'visible' if self.feedback_visible else 'hidden'}")
    
    def update_feedback_interval(self, *args):
        """Update the feedback interval"""
        interval = self.feedback_interval_var.get()
        self.feedback_interval_label.config(text=f"{interval:.1f}s")
        
        if self.ai_coach:
            self.ai_coach.feedback_interval = interval
            print(f"Feedback interval set to {interval:.1f} seconds")

    def create_control_panel(self):
        """Create the control panel with all UI elements"""
        # 3D View controls
        view_frame = ttk.LabelFrame(self.controls, text="3D View Controls")
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        view_buttons_frame = ttk.Frame(view_frame)
        view_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(view_buttons_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(view_buttons_frame, text="Toggle Background", command=self.toggle_background).pack(side=tk.LEFT, padx=5)
        ttk.Button(view_buttons_frame, text="Rotate 90°", command=self.rotate_90_degrees).pack(side=tk.LEFT, padx=5)
        
        # Sensor connection frame
        sensor_conn_frame = ttk.LabelFrame(self.controls, text="Sensor Connection")
        sensor_conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        sensor_entry_frame = ttk.Frame(sensor_conn_frame)
        sensor_entry_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sensor_entry_frame, text="IP:").pack(side=tk.LEFT, padx=5)
        self.ip_entry = ttk.Entry(sensor_entry_frame, width=15)
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        self.ip_entry.insert(0, "192.168.0.14")
        
        ttk.Label(sensor_entry_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = ttk.Entry(sensor_entry_frame, width=6)
        self.port_entry.pack(side=tk.LEFT, padx=5)
        self.port_entry.insert(0, "80")
        
        sensor_button_frame = ttk.Frame(sensor_conn_frame)
        sensor_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(sensor_button_frame, text="Connect", command=self.connect_sensor).pack(side=tk.LEFT, padx=5)
        ttk.Button(sensor_button_frame, text="Reconnect", command=self.reconnect_sensor).pack(side=tk.LEFT, padx=5)
        
        # Add advanced settings toggle
        adv_settings_button = ttk.Button(sensor_conn_frame, text="Show Advanced Settings", command=self.toggle_sensor_settings)
        adv_settings_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Advanced sensor settings frame (initially hidden)
        self.adv_sensor_frame = ttk.Frame(sensor_conn_frame)
        
        # Auto reconnect checkbox
        self.auto_reconnect_var = tk.BooleanVar(value=True)
        auto_reconnect_check = ttk.Checkbutton(self.adv_sensor_frame, text="Auto Reconnect", 
                                            variable=self.auto_reconnect_var,
                                            command=self.update_reconnect_setting)
        auto_reconnect_check.pack(fill=tk.X, padx=5, pady=2)
        
        # Timeout slider
        timeout_frame = ttk.Frame(self.adv_sensor_frame)
        timeout_frame.pack(fill=tk.X, pady=2)
        ttk.Label(timeout_frame, text="Timeout (s):").pack(side=tk.LEFT, padx=5)
        self.timeout_var = tk.DoubleVar(value=3.0)
        timeout_scale = ttk.Scale(timeout_frame, from_=0.5, to=10.0, 
                                variable=self.timeout_var, 
                                command=self.update_timeout)
        timeout_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.timeout_label = ttk.Label(timeout_frame, text="3.0")
        self.timeout_label.pack(side=tk.LEFT, padx=5)
        
        # Check interval slider
        interval_frame = ttk.Frame(self.adv_sensor_frame)
        interval_frame.pack(fill=tk.X, pady=2)
        ttk.Label(interval_frame, text="Check Interval (s):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.DoubleVar(value=1.0)
        interval_scale = ttk.Scale(interval_frame, from_=0.1, to=5.0, 
                                 variable=self.interval_var, 
                                 command=self.update_check_interval)
        interval_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.interval_label = ttk.Label(interval_frame, text="1.0")
        self.interval_label.pack(side=tk.LEFT, padx=5)
        
        # Max retries
        retry_frame = ttk.Frame(self.adv_sensor_frame)
        retry_frame.pack(fill=tk.X, pady=2)
        ttk.Label(retry_frame, text="Max Retries:").pack(side=tk.LEFT, padx=5)
        self.max_retries_var = tk.IntVar(value=3)
        retry_values = [1, 2, 3, 5, 10]
        retry_combo = ttk.Combobox(retry_frame, values=retry_values, 
                                 textvariable=self.max_retries_var, 
                                 width=5, state="readonly")
        retry_combo.pack(side=tk.LEFT, padx=5)
        retry_combo.bind("<<ComboboxSelected>>", lambda e: self.update_max_retries())
        
        # Data points to display
        display_frame = ttk.Frame(self.adv_sensor_frame)
        display_frame.pack(fill=tk.X, pady=2)
        ttk.Label(display_frame, text="Display Points:").pack(side=tk.LEFT, padx=5)
        self.data_points_var = tk.IntVar(value=50)
        points_values = [10, 20, 50, 100, 200]
        points_combo = ttk.Combobox(display_frame, values=points_values, 
                                  textvariable=self.data_points_var, 
                                  width=5, state="readonly")
        points_combo.pack(side=tk.LEFT, padx=5)
        points_combo.bind("<<ComboboxSelected>>", lambda e: self.update_data_points())
        
        # Action Classification controls
        action_control_frame = ttk.LabelFrame(self.controls, text="Action Classification Controls")
        action_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Detection rate slider
        rate_frame = ttk.Frame(action_control_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Detection Rate (s):").pack(side=tk.LEFT, padx=5)
        self.detection_rate_var = tk.DoubleVar(value=3.0)  # Default 3.0 seconds
        rate_scale = ttk.Scale(rate_frame, from_=0.5, to=10.0, 
                              variable=self.detection_rate_var, 
                              command=self.update_detection_rate)
        rate_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.detection_rate_label = ttk.Label(rate_frame, text="3.0")
        self.detection_rate_label.pack(side=tk.LEFT, padx=5)
        
        # Increase/decrease buttons
        rate_btn_frame = ttk.Frame(action_control_frame)
        rate_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(rate_btn_frame, text="Faster Detection", 
                 command=self.increase_detection_rate).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(rate_btn_frame, text="Slower Detection", 
                 command=self.decrease_detection_rate).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Confidence threshold slider
        threshold_frame = ttk.Frame(action_control_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text="Confidence Threshold (%):").pack(side=tk.LEFT, padx=5)
        self.confidence_threshold_var = tk.DoubleVar(value=20.0)  # Default 20%
        threshold_scale = ttk.Scale(threshold_frame, from_=5.0, to=95.0, 
                                  variable=self.confidence_threshold_var, 
                                  command=self.update_confidence_threshold)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.threshold_label = ttk.Label(threshold_frame, text="20.0")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # AI Coaching controls
        if self.coach_available:
            coaching_frame = ttk.LabelFrame(self.controls, text="AI Coaching")
            coaching_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Exercise selection
            exercise_frame = ttk.Frame(coaching_frame)
            exercise_frame.pack(fill=tk.X, pady=2)
            ttk.Label(exercise_frame, text="Exercise:").pack(side=tk.LEFT, padx=5)
            self.exercise_var = tk.StringVar(value=SUPPORTED_EXERCISES[0] if SUPPORTED_EXERCISES else "None")
            exercise_combo = ttk.Combobox(exercise_frame, values=SUPPORTED_EXERCISES, 
                                        textvariable=self.exercise_var, 
                                        width=15, state="readonly")
            exercise_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            exercise_combo.bind("<<ComboboxSelected>>", self.on_exercise_selected)
            
            # Start/end exercise buttons
            exercise_btn_frame = ttk.Frame(coaching_frame)
            exercise_btn_frame.pack(fill=tk.X, pady=2)
            ttk.Button(exercise_btn_frame, text="Start Exercise", 
                     command=self.start_exercise).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            ttk.Button(exercise_btn_frame, text="End Exercise", 
                     command=self.end_exercise).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Feedback display toggle
            feedback_frame = ttk.Frame(coaching_frame)
            feedback_frame.pack(fill=tk.X, pady=2)
            self.feedback_visible_var = tk.BooleanVar(value=True)
            feedback_check = ttk.Checkbutton(feedback_frame, text="Show Feedback", 
                                         variable=self.feedback_visible_var,
                                         command=self.toggle_feedback_display)
            feedback_check.pack(side=tk.LEFT, padx=5)
            
            # Feedback interval slider
            interval_frame = ttk.Frame(coaching_frame)
            interval_frame.pack(fill=tk.X, pady=2)
            ttk.Label(interval_frame, text="Feedback Interval (s):").pack(side=tk.LEFT, padx=5)
            self.feedback_interval_var = tk.DoubleVar(value=3.0)  # Default 3.0 seconds
            feedback_scale = ttk.Scale(interval_frame, from_=1.0, to=10.0, 
                                     variable=self.feedback_interval_var, 
                                     command=self.update_feedback_interval)
            feedback_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.feedback_interval_label = ttk.Label(interval_frame, text="3.0")
            self.feedback_interval_label.pack(side=tk.LEFT, padx=5)
            
            # Add Analysis Data button (right below AI coaching section)
            analysis_frame = ttk.LabelFrame(self.controls, text="Motion Analysis")
            analysis_frame.pack(fill=tk.X, padx=5, pady=5)
            
            analysis_button = ttk.Button(analysis_frame, text="Show Analysis Data", 
                                      command=self.show_analysis_options,
                                      style='Modern.TButton')
            analysis_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Additional control sections can be added here
        
        # Update scrollable region
        self.controls.update_idletasks()
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        
        # Bind mouse wheel to scrolling
        self.control_canvas.bind_all("<MouseWheel>", lambda e: self.control_canvas.yview_scroll(-1*(e.delta//120), "units"))
    
    def show_analysis_options(self):
        """Show a popup with analysis options"""
        if self.analysis_window and self.analysis_window.winfo_exists():
            self.analysis_window.lift()
            return
            
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title("Motion Analysis")
        self.analysis_window.geometry("300x350")
        self.analysis_window.transient(self.root)
        self.analysis_window.resizable(False, False)
        
        # When window is closed, reset analysis_window to None
        self.analysis_window.protocol("WM_DELETE_WINDOW", self.close_analysis_window)
        
        # Add options
        ttk.Label(self.analysis_window, text="Select Analysis Type:", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        # Cosine Similarity button
        cosine_btn = ttk.Button(self.analysis_window, text="Cosine Similarity", 
                             command=lambda: self.show_analysis_data("cosine_similarity"))
        cosine_btn.pack(fill=tk.X, padx=20, pady=5)
        
        # DTW button
        dtw_btn = ttk.Button(self.analysis_window, text="Dynamic Time Warping (DTW)", 
                          command=lambda: self.show_analysis_data("dtw"))
        dtw_btn.pack(fill=tk.X, padx=20, pady=5)
        
        # Fourier Analysis button
        fourier_btn = ttk.Button(self.analysis_window, text="Fourier Analysis", 
                               command=lambda: self.show_analysis_data("fourier"))
        fourier_btn.pack(fill=tk.X, padx=20, pady=5)
        
        # Create frame for analysis data display
        self.analysis_frame = ttk.Frame(self.analysis_window)
        self.analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def close_analysis_window(self):
        """Close the analysis window"""
        if self.analysis_window:
            self.analysis_window.destroy()
            self.analysis_window = None
    
    def show_analysis_data(self, analysis_type):
        """Show the selected analysis data"""
        self.analysis_type = analysis_type
        
        # Clear previous content
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        
        # Get analysis data
        data = self.calculate_analysis_data(analysis_type)
        
        # Create matplotlib figure
        fig = plt.Figure(figsize=(4, 3), dpi=80)
        ax = fig.add_subplot(111)
        
        if analysis_type == "cosine_similarity":
            ax.set_title("Movement Similarity Over Time")
            ax.set_ylabel("Similarity (0-1)")
            ax.plot(data, color='blue')
            ax.set_ylim(0, 1)
            ax.grid(True)
            
            # Show statistics
            mean_val = np.mean(data) if len(data) > 0 else 0
            ttk.Label(self.analysis_frame, 
                    text=f"Average Similarity: {mean_val:.2f}\n"
                        f"Higher values indicate more consistent movement.").pack(pady=5)
                        
        elif analysis_type == "dtw":
            ax.set_title("Form Deviation Over Time")
            ax.set_ylabel("DTW Distance")
            ax.plot(data, color='red')
            ax.grid(True)
            
            # Show statistics
            mean_val = np.mean(data) if len(data) > 0 else 0
            ttk.Label(self.analysis_frame, 
                    text=f"Average Deviation: {mean_val:.2f}\n"
                        f"Lower values indicate better form.").pack(pady=5)
                        
        elif analysis_type == "fourier":
            ax.set_title("Movement Frequency Analysis")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Magnitude")
            
            if len(data[0]) > 0:
                # Show only positive frequencies and first half (symmetric)
                pos_idx = len(data[0]) // 2
                freqs = data[0][:pos_idx]
                mags = data[1][:pos_idx]
                ax.plot(freqs, mags, color='green')
                
                # Find dominant frequency
                dom_idx = np.argmax(mags)
                dom_freq = freqs[dom_idx]
                ttk.Label(self.analysis_frame, 
                        text=f"Dominant Frequency: {dom_freq:.3f} Hz\n"
                            f"Rhythm consistency: {np.max(mags)/np.sum(mags):.2f}").pack(pady=5)
            else:
                ttk.Label(self.analysis_frame, text="Not enough data for analysis.").pack(pady=5)
                
        # Create canvas for matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Send analysis data to AI coach for feedback
        if hasattr(self, 'ai_coach') and self.ai_coach and self.exercise_active:
            self.send_analysis_to_ai_coach(analysis_type, data)
    
    def calculate_analysis_data(self, analysis_type):
        """Calculate or retrieve the requested analysis data"""
        try:
            if analysis_type == "cosine_similarity":
                # Use data from Pose3D visualizer or generate new
                if hasattr(self.pose_estimator.pose_3d, 'cosine_similarities') and len(self.pose_estimator.pose_3d.cosine_similarities) > 0:
                    self.last_analysis_data['cosine_similarity'] = self.pose_estimator.pose_3d.cosine_similarities.copy()
                elif len(self.last_analysis_data['cosine_similarity']) == 0:
                    # Generate sample data if no data available yet
                    self.last_analysis_data['cosine_similarity'] = np.random.uniform(0.7, 0.95, 100)
                return self.last_analysis_data['cosine_similarity']
                
            elif analysis_type == "dtw":
                # Use data from Pose3D visualizer or generate new
                if hasattr(self.pose_estimator.pose_3d, 'dtw_distances') and len(self.pose_estimator.pose_3d.dtw_distances) > 0:
                    self.last_analysis_data['dtw'] = self.pose_estimator.pose_3d.dtw_distances.copy()
                elif len(self.last_analysis_data['dtw']) == 0:
                    # Generate sample data if no data available yet
                    self.last_analysis_data['dtw'] = np.random.uniform(0.1, 2.0, 100)
                return self.last_analysis_data['dtw']
                
            elif analysis_type == "fourier":
                # Use data from Pose3D visualizer or generate new
                if hasattr(self.pose_estimator.pose_3d, 'frequency_data'):
                    self.last_analysis_data['fourier'] = [
                        self.pose_estimator.pose_3d.frequency_data['frequencies'],
                        self.pose_estimator.pose_3d.frequency_data['magnitudes']
                    ]
                elif len(self.last_analysis_data['fourier']) == 0:
                    # Generate sample data if no data available yet
                    x = np.linspace(0, 1, 100)
                    y = np.abs(np.fft.fft(np.sin(2 * np.pi * 5 * x) + 0.5 * np.sin(2 * np.pi * 10 * x)))
                    self.last_analysis_data['fourier'] = [x, y]
                return self.last_analysis_data['fourier']
        except Exception as e:
            print(f"Error calculating analysis data: {e}")
            return [] if analysis_type != "fourier" else [[], []]
    
    def send_analysis_to_ai_coach(self, analysis_type, data):
        """Send the analysis data to AI coach for feedback"""
        try:
            if not hasattr(self, 'ai_coach') or not self.ai_coach or not self.exercise_active:
                return
                
            # Format the data for AI coach
            analysis_summary = {}
            
            if analysis_type == "cosine_similarity":
                if len(data) > 0:
                    analysis_summary = {
                        'type': 'cosine_similarity',
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'min': float(np.min(data)),
                        'max': float(np.max(data))
                    }
            
            elif analysis_type == "dtw":
                if len(data) > 0:
                    analysis_summary = {
                        'type': 'dtw',
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'min': float(np.min(data)),
                        'max': float(np.max(data))
                    }
            
            elif analysis_type == "fourier":
                if len(data[0]) > 0:
                    # Find dominant frequency
                    pos_idx = len(data[0]) // 2
                    freqs = data[0][:pos_idx]
                    mags = data[1][:pos_idx]
                    dom_idx = np.argmax(mags)
                    dom_freq = float(freqs[dom_idx]) if dom_idx < len(freqs) else 0
                    
                    analysis_summary = {
                        'type': 'fourier',
                        'dominant_frequency': dom_freq,
                        'rhythm_consistency': float(np.max(mags)/np.sum(mags)) if np.sum(mags) > 0 else 0
                    }
            
            # Send to AI coach if we have data
            if analysis_summary and hasattr(self.ai_coach, 'add_analysis_data'):
                self.ai_coach.add_analysis_data(analysis_summary)
        
        except Exception as e:
            print(f"Error sending analysis to AI coach: {e}")

    def rotate_90_degrees(self):
        """Rotate the 3D view by 90 degrees around the Y axis"""
        try:
            if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'pose_3d'):
                # Get the current rotation
                current_y_rotation = self.pose_estimator.pose_3d.camera_rotation[1]
                # Add 90 degrees (convert to radians)
                new_y_rotation = current_y_rotation + 90.0 * np.pi / 180.0
                # Update the rotation
                self.pose_estimator.pose_3d.camera_rotation[1] = new_y_rotation
                print("Rotated view 90 degrees around Y axis")
        except Exception as e:
            print(f"Error rotating view: {e}")

# PoseActionClassifier - Direct implementation from realtime_pose_detection.py
class PoseActionClassifier:
    def __init__(self, model_dir='model_output/model_output/'):
        self.model_dir = model_dir
        self.model_ready = False
        
        try:
            # Load preprocessing information
            with open(os.path.join(model_dir, 'preprocess_info.json'), 'r') as f:
                self.preprocess_info = json.load(f)
            
            # Load label map
            with open(os.path.join(model_dir, 'label_map.json'), 'r') as f:
                self.label_map = json.load(f)
            
            # Get model parameters
            self.input_shape = self.preprocess_info['input_shape']
            self.frames_per_video = self.preprocess_info['frames_per_video']
            self.img_size = tuple(self.preprocess_info['img_size'])
            self.classes = self.preprocess_info['classes']
            
            print(f"Action Classifier - Input shape: {self.input_shape}")
            print(f"Action Classifier - Classes: {self.classes}")
            
            # Initialize the frame buffer and prediction history
            self.frame_buffer = collections.deque(maxlen=self.frames_per_video)
            self.prediction_history = collections.deque(maxlen=5)
            self.last_prediction_time = 0
            self.prediction_interval = 3.0  # Update prediction every 3 seconds
            
            # Build and load the model
            self.build_and_load_model()
            
        except Exception as e:
            print(f"Error initializing PoseActionClassifier: {str(e)}")
            self.model_ready = False
    
    def build_and_load_model(self):
        """Build the model and load weights"""
        try:
            print("\nBuilding action classification model...")
            model_input_shape = (self.frames_per_video, self.img_size[0], self.img_size[1], 3)
            self.model = self.build_model(model_input_shape, len(self.classes))
            
            # Try to load pre-trained weights
            weights_loaded = False
            
            # First try: directly load weights from .h5 file
            weights_file = os.path.join(self.model_dir, 'model.h5')
            if os.path.exists(weights_file):
                try:
                    print(f"Loading weights from {weights_file}...")
                    self.model.load_weights(weights_file)
                    print("Weights loaded successfully!")
                    weights_loaded = True
                except Exception as e:
                    print(f"Could not load weights from {weights_file}: {e}")
            
            # Second try: load from dedicated weights file
            if not weights_loaded:
                weights_file = os.path.join(self.model_dir, 'model_weights.weights.h5')
                if os.path.exists(weights_file):
                    try:
                        print(f"Loading weights from {weights_file}...")
                        self.model.load_weights(weights_file)
                        print("Weights loaded successfully!")
                        weights_loaded = True
                    except Exception as e:
                        print(f"Could not load weights from {weights_file}: {e}")
            
            if weights_loaded:
                self.model_ready = True
                print("Action classification model ready for inference.")
            else:
                print("\nWARNING: No weights were loaded. The model will not produce accurate predictions.")
                self.model_ready = False
                
        except Exception as e:
            print(f"\nError building/loading action classification model: {e}")
            self.model_ready = False
    
    def build_model(self, input_shape, num_classes):
        """Build the model with the exact architecture as in training"""
        from tensorflow.keras.models import Sequential
        
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
            return "Unknown", 0.0
        
        # Count occurrences of each class
        class_counts = {}
        for pred, conf in self.prediction_history:
            if pred in class_counts:
                class_counts[pred].append(conf)
            else:
                class_counts[pred] = [conf]
        
        # Find the most common class
        if class_counts:
            majority_class = max(class_counts.keys(), key=lambda k: len(class_counts[k]))
            avg_confidence = sum(class_counts[majority_class]) / len(class_counts[majority_class])
            return majority_class, avg_confidence * 100  # Convert to percentage
        else:
            return "Unknown", 0.0
    
    def predict(self, frame):
        """Process a frame and make a prediction when conditions are met."""
        if not self.model_ready:
            return "Unknown", 0.0
            
        # Add frame to buffer
        self.add_frame(frame)
        
        # Return if buffer not full
        if not self.is_buffer_full():
            return "Unknown", 0.0
        
        # Check if it's time for a new prediction
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            # Return the last stable prediction
            return self.get_majority_prediction()
        
        try:
            # Stack frames into a batch - exactly as in training
            frames_array = np.array(list(self.frame_buffer))
            batch = np.expand_dims(frames_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(batch, verbose=0)
            
            # Get the predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Map index to class name
            predicted_class = self.classes[predicted_class_idx]
            
            # Add to prediction history
            self.prediction_history.append((predicted_class, confidence))
            self.last_prediction_time = current_time
            
            # Return smoothed prediction
            return self.get_majority_prediction()
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Unknown", 0.0

    def reconnect_sensor(self):
        """Attempt to reconnect to the last known sensor"""
        try:
            if hasattr(self, 'last_known_ip') and hasattr(self, 'last_known_port'):
                if self.last_known_ip and self.last_known_port:
                    print(f"Attempting to reconnect to sensor at {self.last_known_ip}:{self.last_known_port}")
                    
                    # Update the IP and port entries for user feedback
                    self.ip_entry.delete(0, tk.END)
                    self.ip_entry.insert(0, self.last_known_ip)
                    
                    self.port_entry.delete(0, tk.END)
                    self.port_entry.insert(0, str(self.last_known_port))
                    
                    # Reset connection status
                    self.connection_retry_count = 0
                    self.sensor_ip = self.last_known_ip
                    self.sensor_port = self.last_known_port
                    
                    # Update UI status
                    self.connection_status.delete("all")
                    self.connection_status.create_oval(0, 0, 20, 20, fill='orange')
                    
                    # Force an immediate connection check
                    self.last_sensor_check = 0
                    self.update_sensor_data()
                    
                    return True
                else:
                    print("No previous connection details available")
            else:
                print("No previous connection details available")
                
            return False
        except Exception as e:
            print(f"Error reconnecting to sensor: {e}")
            return False
            
    def update_reconnect_setting(self):
        """Update auto reconnect setting"""
        self.auto_reconnect = self.auto_reconnect_var.get()
        print(f"Auto reconnect {'enabled' if self.auto_reconnect else 'disabled'}")
        
    def update_timeout(self, *args):
        """Update connection timeout value"""
        try:
            timeout = self.timeout_var.get()
            self.timeout_label.config(text=f"{timeout:.1f}")
            print(f"Connection timeout set to {timeout:.1f} seconds")
        except Exception as e:
            print(f"Error updating timeout: {e}")
            
    def update_check_interval(self, *args):
        """Update sensor check interval"""
        try:
            interval = self.interval_var.get()
            self.interval_label.config(text=f"{interval:.1f}")
            self.sensor_check_interval = interval
            print(f"Sensor check interval set to {interval:.1f} seconds")
        except Exception as e:
            print(f"Error updating check interval: {e}")
            
    def update_max_retries(self):
        """Update the maximum retry count"""
        retries = self.max_retries_var.get()
        self.max_retries = retries
        print(f"Maximum connection retries set to {retries}")
        
    def update_data_points(self):
        """Update the number of data points to display"""
        try:
            points = self.data_points_var.get()
            # Store in a class variable for use in sensor data processing
            self.max_sensor_points = points
            print(f"Sensor data display points set to {points}")
        except Exception as e:
            print(f"Error updating data points: {e}")

def main():
    # Initialize and run the application
    app = PoseVisualizerGUI()
    
    # Store the last detected action and confidence
    app.last_action = "Unknown"
    app.last_confidence = 0
    
    # Add update method to the app
    def update():
        """Update the application with new frames"""
        try:
            ret, frame = app.cap.read()
            if ret:
                # Process frame with pose estimator
                processed_frame, landmarks, landmark_world = app.pose_estimator.process_frame(frame)
                
                # Update the input feed - fill the entire canvas
                canvas_width = app.video_canvas.winfo_width() or 288
                canvas_height = app.video_canvas.winfo_height() or 192
                
                # Resize frame to fill the canvas while maintaining aspect ratio
                frame_aspect = frame.shape[1] / frame.shape[0]
                canvas_aspect = canvas_width / canvas_height
                
                if frame_aspect > canvas_aspect:
                    # Frame is wider than canvas, fit to height
                    new_height = canvas_height
                    new_width = int(new_height * frame_aspect)
                else:
                    # Frame is taller than canvas, fit to width
                    new_width = canvas_width
                    new_height = int(new_width / frame_aspect)
                
                input_resized = cv2.resize(processed_frame, (new_width, new_height))
                input_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)))
                app.video_canvas.delete("all")
                app.video_canvas.create_image(canvas_width//2, canvas_height//2, image=input_photo, anchor=tk.CENTER)
                app.video_canvas.image = input_photo
                
                # Create 2D skeleton on black background
                skeleton_width = app.skeleton_canvas.winfo_width() or 288
                skeleton_height = app.skeleton_canvas.winfo_height() or 192
                
                # Create a black image for the skeleton
                skeleton_img = np.zeros((skeleton_height, skeleton_width, 3), dtype=np.uint8)
                
                if landmarks:
                    # Draw the skeleton in white on black background
                    mp_drawing = mp.solutions.drawing_utils
                    mp_pose = mp.solutions.pose
                    
                    # Convert landmarks to the proper format
                    landmark_list = landmark_pb2.NormalizedLandmarkList()
                    for idx, landmark in enumerate(landmarks):
                        landmark_list.landmark.add(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z if hasattr(landmark, 'z') else 0.0,
                            visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                        )
                    
                    # Draw the skeleton in white
                    mp_drawing.draw_landmarks(
                        skeleton_img, 
                        landmark_list, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # Update the 3D pose
                    app.pose_estimator.pose_3d.render_pose(landmark_world or landmarks)
                    
                    # Add landmarks to AI coach if active and available
                    if app.coach_available and app.exercise_active:
                        app.ai_coach.add_landmark_data(landmarks)
                    
                    # Add frame to action classifier if it's ready
                    if hasattr(app, 'action_classifier') and app.action_classifier:
                        app.action_classifier.add_frame(processed_frame)
                        
                        # Check if we can predict action
                        if app.action_classifier.is_buffer_full():
                            action, confidence = app.action_classifier.predict(processed_frame)
                            
                            # Get threshold from UI settings
                            threshold = app.confidence_threshold_var.get() if hasattr(app, 'confidence_threshold_var') else 20.0
                            
                            if action != "Unknown" and confidence > threshold:  # Check against threshold
                                print(f"Detected {action} with confidence {confidence:.1f}%")
                                
                                # Store the last action and confidence
                                app.last_action = action
                                app.last_confidence = confidence
                                
                                # Update action on 3D interface
                                app.pose_estimator.pose_3d.current_action = action
                                app.pose_estimator.pose_3d.action_confidence = confidence
                                
                                # Check for rep counting if using AI coach
                                if app.coach_available and app.exercise_active:
                                    if app.ai_coach.count_rep(landmarks, action):
                                        print(f"Rep counted for {action}!")
                            else:
                                # If confidence is below threshold, show "No Action"
                                if hasattr(app, 'last_action'):
                                    # Only update if no action or different action
                                    if app.last_action != "No Action" or confidence > 0:
                                        print(f"No action detected (below threshold of {threshold}%)")
                                        app.last_action = "No Action"
                                        app.last_confidence = 0
                                        app.pose_estimator.pose_3d.current_action = "No Action"
                                        app.pose_estimator.pose_3d.action_confidence = 0
                
                # Update the skeleton visualization
                skeleton_photo = ImageTk.PhotoImage(image=Image.fromarray(skeleton_img))
                app.skeleton_canvas.delete("all")
                app.skeleton_canvas.create_image(skeleton_width//2, skeleton_height//2, image=skeleton_photo, anchor=tk.CENTER)
                app.skeleton_canvas.image = skeleton_photo
                
                # Update 3D visualization
                if hasattr(app, 'pose_estimator') and hasattr(app.pose_estimator, 'pose_3d'):
                    try:
                        # Get the rendered pygame surface
                        pygame_surface = pygame.surfarray.array3d(app.pose_estimator.pose_3d.screen)
                        pygame_surface = pygame_surface.swapaxes(0, 1)
                        
                        # Get canvas dimensions
                        viz_width = app.colored_frame.winfo_width() or 1280
                        viz_height = app.colored_frame.winfo_height() or 720
                        
                        # Resize to match the canvas dimensions
                        pygame_surface_resized = cv2.resize(pygame_surface, (viz_width, viz_height))
                        colored_photo = ImageTk.PhotoImage(image=Image.fromarray(pygame_surface_resized))
                        
                        app.colored_frame.delete("all")
                        app.colored_frame.create_image(viz_width//2, viz_height//2, image=colored_photo, anchor=tk.CENTER)
                        app.colored_frame.image = colored_photo
                        
                        # Always draw the action classification with the last known valid action
                        if hasattr(app, 'last_action') and app.last_action != "Unknown" and app.last_confidence > 0:
                            draw_action_classification(app, app.last_action, app.last_confidence)
                        
                        # Update AI coach feedback if available
                        if app.coach_available and app.exercise_active:
                            current_time = time.time()
                            
                            # Get sensor data for the AI coach
                            if hasattr(app, 'sensor_data'):
                                # Create a snapshot of current sensor data
                                sensor_snapshot = {
                                    'gyro': {
                                        'x': app.sensor_data['gyro']['x'][-1] if app.sensor_data['gyro']['x'] else 0,
                                        'y': app.sensor_data['gyro']['y'][-1] if app.sensor_data['gyro']['y'] else 0,
                                        'z': app.sensor_data['gyro']['z'][-1] if app.sensor_data['gyro']['z'] else 0
                                    },
                                    'accel': {
                                        'x': app.sensor_data['accel']['x'][-1] if app.sensor_data['accel']['x'] else 0,
                                        'y': app.sensor_data['accel']['y'][-1] if app.sensor_data['accel']['y'] else 0,
                                        'z': app.sensor_data['accel']['z'][-1] if app.sensor_data['accel']['z'] else 0
                                    }
                                }
                                
                                # Add sensor data to AI coach
                                app.ai_coach.add_sensor_data('gyro', sensor_snapshot['gyro'])
                                app.ai_coach.add_sensor_data('accel', sensor_snapshot['accel'])
                            
                            # Get feedback at specified intervals
                            if current_time - app.last_feedback_time >= app.ai_coach.feedback_interval:
                                app.last_feedback_time = current_time
                                
                                # Get feedback from AI coach
                                feedback = app.ai_coach.get_feedback(
                                    app.last_action,
                                    landmarks,
                                    sensor_snapshot if 'sensor_snapshot' in locals() else None,
                                    app.last_confidence
                                )
                                
                                if feedback:
                                    app.current_feedback = feedback
                        
                        # Draw AI coach feedback if available
                        if app.coach_available:
                            draw_coach_feedback(
                                app,
                                app.current_feedback if hasattr(app, 'current_feedback') else None,
                                landmarks,
                                app.sensor_data if hasattr(app, 'sensor_data') else None
                            )
                    except Exception as e:
                        print(f"Error updating 3D visualization: {e}")
                
                # Update sensor data - only use actual sensor data, never simulate
                current_time = time.time()
                
                # Only try to read from sensor if we're connected to avoid simulated data
                if hasattr(app, 'sensor_ip') and app.sensor_ip is not None:
                    app.update_sensor_data()  # This will fetch actual data from the sensor
                else:
                    # No sensor connected - display message
                    app.connection_status.delete("all")
                    app.connection_status.create_oval(0, 0, 20, 20, fill='red')
                
                # Update sensor visualization
                update_sensor_visualization(app)
                
                # Update 3D sensor visualization canvas (previously future canvas)
                update_future_canvas(app)
                
                # Update motion analysis overlay
                app.update_motion_analysis_overlay()
                
                # Schedule next update
                app.root.after(15, update)
        except Exception as e:
            print(f"Error in update loop: {e}")
            # Handle errors gracefully and try to continue
            app.root.after(100, update)
    
    def update_future_canvas(app):
        """Update the sensor 3D visualization canvas with sensor data in 3D"""
        try:
            # Clear the canvas
            app.sensor_3d_canvas.delete("all")
            
            # Set canvas dimensions
            width = app.sensor_3d_canvas.winfo_width() or 288
            height = app.sensor_3d_canvas.winfo_height() or 192
            
            # Create black background
            app.sensor_3d_canvas.create_rectangle(0, 0, width, height, fill="#000000", outline="")
            
            # Add title
            app.sensor_3d_canvas.create_text(width//2, 15, text="Sensor Data 3D", 
                                           fill="white", font=("Arial", 10, "bold"))
            
            # Check if we have sensor data to visualize
            if not hasattr(app, 'sensor_data') or not app.sensor_data['gyro']['x'] or not app.sensor_data['accel']['x']:
                app.sensor_3d_canvas.create_text(width//2, height//2, text="Waiting for sensor data...", fill="white")
                return
                
            # Draw coordinate system
            center_x = width // 2
            center_y = height // 2
            axis_length = 50
            
            # X axis (red)
            app.sensor_3d_canvas.create_line(center_x, center_y, center_x + axis_length, center_y, fill="red", width=2, arrow=tk.LAST)
            app.sensor_3d_canvas.create_text(center_x + axis_length + 10, center_y, text="X", fill="red")
            
            # Y axis (green)
            app.sensor_3d_canvas.create_line(center_x, center_y, center_x, center_y - axis_length, fill="green", width=2, arrow=tk.LAST)
            app.sensor_3d_canvas.create_text(center_x, center_y - axis_length - 10, text="Y", fill="green")
            
            # Z axis (blue) - shown as a perspective line
            app.sensor_3d_canvas.create_line(center_x, center_y, center_x - axis_length//2, center_y - axis_length//2, fill="blue", width=2, arrow=tk.LAST)
            app.sensor_3d_canvas.create_text(center_x - axis_length//2 - 10, center_y - axis_length//2 - 10, text="Z", fill="blue")
            
            # Get the most recent sensor data
            try:
                # Get the last valid gyro and accel data
                gyro_x = app.sensor_data['gyro']['x'][-1] if app.sensor_data['gyro']['x'] else 0
                gyro_y = app.sensor_data['gyro']['y'][-1] if app.sensor_data['gyro']['y'] else 0
                gyro_z = app.sensor_data['gyro']['z'][-1] if app.sensor_data['gyro']['z'] else 0
                
                accel_x = app.sensor_data['accel']['x'][-1] if app.sensor_data['accel']['x'] else 0
                accel_y = app.sensor_data['accel']['y'][-1] if app.sensor_data['accel']['y'] else 0
                accel_z = app.sensor_data['accel']['z'][-1] if app.sensor_data['accel']['z'] else 0
                
                # Normalize and scale vectors for visualization
                gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
                accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                
                # Prevent division by zero
                if gyro_mag > 0:
                    gyro_scale = min(50, max(20, 30 * gyro_mag / 5))  # Scale based on magnitude, but keep within reasonable range
                    gyro_x = gyro_x / gyro_mag * gyro_scale
                    gyro_y = gyro_y / gyro_mag * gyro_scale
                    gyro_z = gyro_z / gyro_mag * gyro_scale
                else:
                    gyro_x, gyro_y, gyro_z = 0, 0, 0
                    
                if accel_mag > 0:
                    accel_scale = min(50, max(20, 30 * accel_mag / 10))  # Scale based on magnitude, but keep within reasonable range
                    accel_x = accel_x / accel_mag * accel_scale
                    accel_y = accel_y / accel_mag * accel_scale
                    accel_z = accel_z / accel_mag * accel_scale
                else:
                    accel_x, accel_y, accel_z = 0, 0, 0
                
                # Draw gyro vector with thick line
                app.sensor_3d_canvas.create_line(
                    center_x, center_y,
                    center_x + gyro_x, center_y - gyro_y,  # Invert Y because canvas coordinates start at top-left
                    fill="yellow", width=3, arrow=tk.LAST
                )
                
                # Draw accel vector with dashed line
                app.sensor_3d_canvas.create_line(
                    center_x, center_y,
                    center_x + accel_x, center_y - accel_y,  # Invert Y because canvas coordinates start at top-left
                    fill="cyan", width=2, dash=(5, 2), arrow=tk.LAST
                )
                
                # Add legend
                legend_y = height - 35
                
                # Gyro legend
                app.sensor_3d_canvas.create_line(center_x - 100, legend_y, center_x - 80, legend_y, fill="yellow", width=3)
                app.sensor_3d_canvas.create_text(center_x - 40, legend_y, text="Gyro", fill="yellow", anchor=tk.W)
                
                # Accel legend
                app.sensor_3d_canvas.create_line(center_x + 10, legend_y, center_x + 30, legend_y, fill="cyan", width=2, dash=(5, 2))
                app.sensor_3d_canvas.create_text(center_x + 45, legend_y, text="Accel", fill="cyan", anchor=tk.W)
                
                # Display values
                app.sensor_3d_canvas.create_text(
                    width - 10, height - 15,
                    text=f"G: {gyro_mag:.2f} rad/s | A: {accel_mag:.2f} m/s²", 
                    fill="white", anchor=tk.E, font=("Arial", 8)
                )
                
            except Exception as e:
                print(f"Error drawing sensor vectors: {e}")
                
        except Exception as e:
            print(f"Error updating sensor 3D visualization: {e}")

    def draw_action_classification(app, action, confidence):
        """Draw action classification information on the 3D interface - positioned below motion analysis"""
        try:
            # Get the 3D canvas
            canvas = app.colored_frame
            
            # Define position for the action classification box
            # Position it below the motion analysis overlay
            overlay_width = 220
            overlay_height = 100
            overlay_x = 10  # Position on the left side
            overlay_y = 316  # Position 1 inch (96px) below original position of 220
            
            # Clear previous overlays
            canvas.delete("action_classification")
            
            # Draw background
            bg_color = "black"
            canvas.create_rectangle(
                overlay_x, overlay_y,
                overlay_x + overlay_width, overlay_y + overlay_height,
                fill=bg_color, stipple='gray50', outline='white',
                tags="action_classification"
            )
            
            # Draw title
            canvas.create_text(
                overlay_x + overlay_width//2, overlay_y + 15,
                text="Action Classification",
                fill='white', font=('Arial', 10, 'bold'),
                tags="action_classification"
            )
            
            # Determine color based on confidence
            if confidence > 80:
                action_color = "#00FF00"  # Green for high confidence
            elif confidence > 50:
                action_color = "#FFFF00"  # Yellow for medium confidence
            else:
                action_color = "#FF0000"  # Red for low confidence
                
            # Display action name
            canvas.create_text(
                overlay_x + overlay_width//2, overlay_y + 45,
                text=action,
                fill=action_color, font=('Arial', 12, 'bold'),
                tags="action_classification"
            )
            
            # Draw confidence text
            canvas.create_text(
                overlay_x + overlay_width//2, overlay_y + 70,
                text=f"Confidence: {confidence:.1f}%",
                fill='white', font=('Arial', 9),
                tags="action_classification"
            )
            
            # Draw confidence meter
            meter_width = 180
            meter_height = 10
            meter_x = overlay_x + (overlay_width - meter_width) // 2
            meter_y = overlay_y + 85
            
            # Draw meter background
            canvas.create_rectangle(
                meter_x, meter_y,
                meter_x + meter_width, meter_y + meter_height,
                fill='gray30', outline='gray50',
                tags="action_classification"
            )
            
            # Draw filled confidence bar
            fill_width = int(meter_width * (confidence / 100))
            canvas.create_rectangle(
                meter_x, meter_y,
                meter_x + fill_width, meter_y + meter_height,
                fill=action_color, outline='',
                tags="action_classification"
            )
        except Exception as e:
            print(f"Error drawing action classification on 3D interface: {e}")
    
    def update_sensor_visualization(app):
        """Update the sensor data visualization"""
        try:
            # Clear the canvas
            app.sensor_canvas.delete("all")
            
            # Get canvas dimensions
            width = app.sensor_canvas.winfo_width() or 288
            height = app.sensor_canvas.winfo_height() or 192
            
            # Create background
            app.sensor_canvas.create_rectangle(0, 0, width, height, fill="#1a1a1a", outline="")
            
            # Add title
            app.sensor_canvas.create_text(width//2, 15, text="Sensor Data", 
                                       fill="white", font=("Arial", 10, "bold"))
            
            if hasattr(app, 'sensor_data'):
                # Use either gyro or accel data based on selected tab
                sensor_type = app.sensor_type.get() if hasattr(app, 'sensor_type') else "gyro"
                data = app.sensor_data[sensor_type]
                
                # Plot line graph if we have enough data points
                if len(data['x']) > 1:
                    # Define graph area
                    margin = 30
                    graph_width = width - 2 * margin
                    graph_height = height - 2 * margin - 20  # Extra space for legend
                    
                    # Draw coordinate axes
                    app.sensor_canvas.create_line(margin, height - margin, 
                                               width - margin, height - margin, 
                                               fill="white", width=1)  # X-axis
                    app.sensor_canvas.create_line(margin, height - margin, 
                                               margin, margin + 20, 
                                               fill="white", width=1)  # Y-axis
                    
                    # Draw axis labels
                    app.sensor_canvas.create_text(width - margin, height - margin + 10, 
                                               text="Time", fill="white", anchor=tk.NE)
                    app.sensor_canvas.create_text(margin - 5, margin + 20, 
                                               text="Value", fill="white", anchor=tk.NE)
                    
                    # Find min/max values for scaling
                    min_val = min([min(data['x'][-100:] or [0]), 
                                  min(data['y'][-100:] or [0]), 
                                  min(data['z'][-100:] or [0])])
                    max_val = max([max(data['x'][-100:] or [0]), 
                                  max(data['y'][-100:] or [0]), 
                                  max(data['z'][-100:] or [0])])
                    
                    # Ensure we have some range
                    if abs(max_val - min_val) < 0.1:
                        max_val = min_val + 1.0
                    
                    # Add padding to min/max
                    value_range = max_val - min_val
                    min_val -= value_range * 0.1
                    max_val += value_range * 0.1
                    
                    # Draw scale
                    app.sensor_canvas.create_text(margin - 5, height - margin, 
                                               text=f"{min_val:.1f}", fill="white", anchor=tk.E)
                    app.sensor_canvas.create_text(margin - 5, margin + 20, 
                                               text=f"{max_val:.1f}", fill="white", anchor=tk.E)
                    
                    # Function to convert data point to canvas coordinates
                    def to_canvas(x_index, y_value):
                        x_canvas = margin + (x_index / (len(data['x']) - 1)) * graph_width
                        # Normalize y value to range [0, 1], then scale to graph height
                        y_normalized = (y_value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                        y_canvas = (height - margin) - y_normalized * graph_height
                        return x_canvas, y_canvas
                    
                    # Draw X component (red)
                    points_x = []
                    for i in range(len(data['x'])):
                        points_x.append(to_canvas(i, data['x'][i]))
                    
                    if len(points_x) > 1:
                        for i in range(len(points_x) - 1):
                            app.sensor_canvas.create_line(points_x[i][0], points_x[i][1], 
                                                       points_x[i+1][0], points_x[i+1][1], 
                                                       fill="red", width=2)
                    
                    # Draw Y component (green)
                    points_y = []
                    for i in range(len(data['y'])):
                        points_y.append(to_canvas(i, data['y'][i]))
                    
                    if len(points_y) > 1:
                        for i in range(len(points_y) - 1):
                            app.sensor_canvas.create_line(points_y[i][0], points_y[i][1], 
                                                       points_y[i+1][0], points_y[i+1][1], 
                                                       fill="green", width=2)
                    
                    # Draw Z component (blue)
                    points_z = []
                    for i in range(len(data['z'])):
                        points_z.append(to_canvas(i, data['z'][i]))
                    
                    if len(points_z) > 1:
                        for i in range(len(points_z) - 1):
                            app.sensor_canvas.create_line(points_z[i][0], points_z[i][1], 
                                                       points_z[i+1][0], points_z[i+1][1], 
                                                       fill="blue", width=2)
                    
                    # Draw legend
                    legend_y = height - 15
                    
                    # X component (red)
                    app.sensor_canvas.create_line(margin + 10, legend_y, margin + 30, legend_y, 
                                               fill="red", width=2)
                    app.sensor_canvas.create_text(margin + 35, legend_y, 
                                               text=f"X: {data['x'][-1]:.2f}", 
                                               fill="white", anchor=tk.W)
                    
                    # Y component (green)
                    app.sensor_canvas.create_line(margin + 100, legend_y, margin + 120, legend_y, 
                                               fill="green", width=2)
                    app.sensor_canvas.create_text(margin + 125, legend_y, 
                                               text=f"Y: {data['y'][-1]:.2f}", 
                                               fill="white", anchor=tk.W)
                    
                    # Z component (blue)
                    app.sensor_canvas.create_line(margin + 190, legend_y, margin + 210, legend_y, 
                                               fill="blue", width=2)
                    app.sensor_canvas.create_text(margin + 215, legend_y, 
                                               text=f"Z: {data['z'][-1]:.2f}", 
                                               fill="white", anchor=tk.W)
                    
                    # Show current magnitude
                    current_mag = data['magnitude'][-1] if data['magnitude'] else 0
                    app.sensor_canvas.create_text(width - margin, margin + 10, 
                                               text=f"Magnitude: {current_mag:.2f}", 
                                               fill="yellow", font=("Arial", 9, "bold"), 
                                               anchor=tk.NE)
                else:
                    # Show message if no data
                    app.sensor_canvas.create_text(width//2, height//2, 
                                               text="Waiting for sensor data...", 
                                               fill="white")
        except Exception as e:
            print(f"Error updating sensor visualization: {e}")

    def draw_coach_feedback(self, feedback, landmarks=None, sensor_data=None):
        """Draw AI coaching feedback on the right side of the 3D interface"""
        try:
            # Get the 3D canvas
            canvas = self.colored_frame
            
            # Define position for the coach feedback box
            feedback_width = 300
            feedback_height = 150
            feedback_x = self.colored_frame.winfo_width() - feedback_width - 10  # Right side
            feedback_y = 10  # Top of the screen
            
            # Clear previous feedback
            canvas.delete("coach_feedback")
            
            # Draw background
            bg_color = "black"
            canvas.create_rectangle(
                feedback_x, feedback_y,
                feedback_x + feedback_width, feedback_y + feedback_height,
                fill=bg_color, stipple='gray50', outline='white',
                tags="coach_feedback"
            )
            
            # Draw title
            canvas.create_text(
                feedback_x + feedback_width//2, feedback_y + 15,
                text="AI Coach Feedback",
                fill='white', font=('Arial', 10, 'bold'),
                tags="coach_feedback"
            )
            
            # Draw feedback text
            if feedback and self.feedback_visible:
                canvas.create_text(
                    feedback_x + feedback_width//2, feedback_y + feedback_height//2 + 10,
                    text=feedback,
                    fill='yellow', font=('Arial', 9),
                    width=feedback_width - 20,  # Text wrapping width
                    tags="coach_feedback"
                )
            else:
                canvas.create_text(
                    feedback_x + feedback_width//2, feedback_y + feedback_height//2 + 10,
                    text="Start an exercise to get AI feedback",
                    fill='yellow', font=('Arial', 9),
                    tags="coach_feedback"
                )
                
            # Add Analysis Data button below the coach feedback box
            btn_width = 150
            btn_height = 25
            btn_x = feedback_x + (feedback_width - btn_width) // 2
            btn_y = feedback_y + feedback_height + 5
            
            # Clear previous button
            canvas.delete("analysis_button")
            
            # Draw button background
            canvas.create_rectangle(
                btn_x, btn_y,
                btn_x + btn_width, btn_y + btn_height,
                fill="#333333", outline='#666666',
                tags="analysis_button"
            )
            
            # Draw button text
            canvas.create_text(
                btn_x + btn_width//2, btn_y + btn_height//2,
                text="Show Analysis Data",
                fill='white', font=('Arial', 9, 'bold'),
                tags="analysis_button"
            )
            
            # Add click binding for the button
            if hasattr(self, 'analysis_button_binding') and self.analysis_button_binding:
                canvas.tag_unbind("analysis_button", "<Button-1>", self.analysis_button_binding)
            
            self.analysis_button_binding = canvas.tag_bind(
                "analysis_button", "<Button-1>", 
                lambda event: self.show_analysis_options() if hasattr(self, 'show_analysis_options') else None
            )
                
        except Exception as e:
            print(f"Error drawing coach feedback: {e}")

    # Start the update loop
    update()
    # Run the main loop
    app.root.mainloop()

if __name__ == "__main__":
    main() 