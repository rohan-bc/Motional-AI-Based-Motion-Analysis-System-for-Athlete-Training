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
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, TimeDistributed, ConvLSTM2D, MaxPooling3D, BatchNormalization, Input
import tensorflow as tf
import collections

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
        
        # Create visualization frames
        input_frame = frame.copy()
        skeleton_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        
        if results.pose_landmarks:
            # Draw skeleton on input frame
            self.mp_draw.draw_landmarks(
                input_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2))
            
            # Draw white skeleton
            self.mp_draw.draw_landmarks(
                skeleton_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2))
            
            # Transform landmarks for 3D visualization
            landmarks_3d = []
            for landmark in results.pose_landmarks.landmark:
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
            
            # Render the 3D pose
            self.pose_3d.render_pose(landmarks_3d)
        else:
            # Render default T-pose when no person is detected
            self.pose_3d.render_pose(None)
        
        # Get the 3D visualization surface
        pygame_surface = pygame.surfarray.array3d(self.pose_3d.screen)
        pygame_surface = pygame_surface.swapaxes(0, 1)
        
        return input_frame, skeleton_frame, pygame_surface

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
        
        # Initialize action classifier using PoseActionClassifier
        try:
            # Use PoseActionClassifier instead of ActionClassifier
            self.action_classifier = PoseActionClassifier()
            
            # Make sure the model is properly loaded
            if not self.action_classifier.model_ready:
                print("Warning: PoseActionClassifier model not ready. Check model files.")
            else:
                print("PoseActionClassifier initialized successfully with model")
            
            # Initialize classification storage
            self.classification_result = None
            
        except Exception as e:
            print(f"Error initializing PoseActionClassifier: {str(e)}")
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
        
        # Create main frame with 2x3 grid
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Configure grid weights for 2x3 layout
        for i in range(3):  # 3 columns
            self.main_frame.grid_columnconfigure(i, weight=1)
        for i in range(2):  # 2 rows
            self.main_frame.grid_rowconfigure(i, weight=1)
        
        # Left column frame for input and 2D skeleton
        left_frame = ttk.Frame(self.main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        # Input feed (top)
        self.input_frame = ttk.Label(left_frame)
        self.input_frame.pack(pady=(0, 5), fill='x')
        ttk.Label(left_frame, text="Input Feed").pack(anchor='w')
        
        # 2D skeleton (below input feed)
        self.skeleton_frame = ttk.Label(left_frame)
        self.skeleton_frame.pack(pady=(0, 5), fill='x')
        ttk.Label(left_frame, text="2D Skeleton").pack(anchor='w')
        
        # 3D visualization (large, spanning middle and right)
        self.colored_frame = tk.Canvas(self.main_frame, bg='black', width=1280, height=720)
        self.colored_frame.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky='nsew')
        ttk.Label(self.main_frame, text="3D Visualization").grid(row=0, column=1, columnspan=2, sticky='n')
        
        # Sensor data (bottom-middle)
        self.sensor_canvas = tk.Canvas(self.main_frame, bg='black')
        self.sensor_canvas.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        ttk.Label(self.main_frame, text="Sensor Data").grid(row=1, column=1, sticky='n')
        
        # Add connection status indicator
        self.connection_status = tk.Canvas(self.sensor_canvas, width=20, height=20, bg='black', highlightthickness=0)
        self.connection_status.place(x=10, y=10)
        self.connection_status.create_oval(0, 0, 20, 20, fill='red')
        
        # Add sensor type selector
        self.sensor_type = tk.StringVar(value="gyro")
        ttk.Radiobutton(self.sensor_canvas, text="Gyroscope", variable=self.sensor_type, value="gyro").place(x=40, y=10)
        ttk.Radiobutton(self.sensor_canvas, text="Accelerometer", variable=self.sensor_type, value="accel").place(x=150, y=10)
        
        # Create scrollable control frame (bottom-right)
        # Create a canvas and scrollbar for the control frame
        self.control_canvas = tk.Canvas(self.main_frame)
        self.control_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.control_canvas.yview)
        self.control_frame = ttk.Frame(self.control_canvas)
        
        # Configure scrolling
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')
        self.control_scrollbar.grid(row=1, column=2, sticky='nse', pady=5)
        
        # Create a frame inside canvas for controls
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor='nw', width=self.control_canvas.winfo_width())
        
        # Add application controls with modern styling
        style = ttk.Style()
        style.configure('Modern.TButton', padding=10, font=('Helvetica', 10))
        style.configure('Modern.TLabel', font=('Helvetica', 10))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(self.control_frame, text="Camera Controls", padding=10)
        camera_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(camera_frame, text="Switch Camera", style='Modern.TButton', command=self.switch_camera).pack(fill='x', padx=5, pady=2)
        
        # View controls
        view_frame = ttk.LabelFrame(self.control_frame, text="View Controls", padding=10)
        view_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(view_frame, text="Toggle Background", style='Modern.TButton', command=self.toggle_background).pack(fill='x', padx=5, pady=2)
        ttk.Button(view_frame, text="Reset View", style='Modern.TButton', command=self.reset_view).pack(fill='x', padx=5, pady=2)
        
        # Sensor controls
        sensor_frame = ttk.LabelFrame(self.control_frame, text="Sensor Controls", padding=10)
        sensor_frame.pack(fill='x', padx=5, pady=5)
        
        # IP input
        ip_frame = ttk.Frame(sensor_frame)
        ip_frame.pack(fill='x', pady=2)
        ttk.Label(ip_frame, text="IP:", style='Modern.TLabel').pack(side='left', padx=5)
        self.ip_entry = ttk.Entry(ip_frame)
        self.ip_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # Port input
        port_frame = ttk.Frame(sensor_frame)
        port_frame.pack(fill='x', pady=2)
        ttk.Label(port_frame, text="Port:", style='Modern.TLabel').pack(side='left', padx=5)
        self.port_entry = ttk.Entry(port_frame)
        self.port_entry.insert(0, "8080")
        self.port_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # Sensor buttons
        ttk.Button(sensor_frame, text="Connect Sensor", style='Modern.TButton', command=self.connect_sensor).pack(fill='x', padx=5, pady=2)
        
        # Application controls
        app_frame = ttk.LabelFrame(self.control_frame, text="Application", padding=10)
        app_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(app_frame, text="Quit", style='Modern.TButton', command=self.quit_app).pack(fill='x', padx=5, pady=2)
        
        # Add 3D control instructions with modern styling
        instructions_frame = ttk.LabelFrame(self.control_frame, text="Controls Help", padding=10)
        instructions_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(
            instructions_frame,
            text="3D Controls:\n\nLeft Mouse Button:\nDrag to Rotate & Pan\n\nMouse Wheel:\nZoom In/Out",
            style='Modern.TLabel',
            justify='left'
        ).pack(pady=5)
        
        # Configure control frame scrolling
        self.control_frame.bind('<Configure>', self.on_frame_configure)
        self.control_canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Initialize mouse position variables
        self.last_x = 0
        self.last_y = 0
        self.is_dragging = False
        
        # Bind mouse events for 3D controls
        self.colored_frame.bind('<Button-1>', self.on_mouse_down)
        self.colored_frame.bind('<B1-Motion>', self.on_mouse_drag)
        self.colored_frame.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.colored_frame.bind('<MouseWheel>', self.on_mouse_wheel)
        
        # Set initial 3D view
        self.pose_estimator.pose_3d.rotation_y = 0
        self.pose_estimator.pose_3d.rotation_x = 0
        
        # Create analysis overlay with increased size
        self.analysis_overlay = tk.Canvas(self.colored_frame, bg='black', highlightthickness=0)
        self.analysis_overlay.place(x=10, y=10, width=300, height=200)  # Reduced size
        self.analysis_overlay.create_rectangle(0, 0, 300, 200, fill='black', stipple='gray50')
        
        # Add motion analysis display in right top area
        self.motion_analysis_overlay = tk.Canvas(self.colored_frame, bg='black', highlightthickness=0)
        self.motion_analysis_overlay.place(x=self.colored_frame.winfo_width()-310, y=10, width=300, height=200)  # Adjusted position and size
        self.motion_analysis_overlay.create_rectangle(0, 0, 300, 200, fill='black', stipple='gray50')
        
        # Add motion analysis display
        self.motion_analysis_frame = ttk.LabelFrame(self.control_frame, text="Motion Analysis", padding=10)
        self.motion_analysis_frame.pack(fill='x', padx=5, pady=5)
        
        # Add motion analysis metrics
        self.motion_metrics = {
            'cosine_similarity': ttk.Label(self.motion_analysis_frame, text="Cosine Similarity: 0.0"),
            'dtw_distance': ttk.Label(self.motion_analysis_frame, text="DTW Distance: 0.0"),
            'repetition_count': ttk.Label(self.motion_analysis_frame, text="Repetitions: 0")
        }
        
        for metric in self.motion_metrics.values():
            metric.pack(fill='x', padx=5, pady=2)
        
        # Add ideal pose capture button
        ttk.Button(self.motion_analysis_frame, text="Capture Ideal Pose", 
                  command=self.capture_ideal_pose).pack(fill='x', padx=5, pady=2)
        
        # Add rotation button to view controls
        ttk.Button(view_frame, text="Rotate 90°", style='Modern.TButton', command=self.rotate_90_degrees).pack(fill='x', padx=5, pady=2)
        
        # Create dropdown menu for motion analysis data
        self.motion_analysis_var = tk.StringVar(value="Data")
        self.motion_analysis_menu = ttk.OptionMenu(self.colored_frame, self.motion_analysis_var, "Data", 
                                                 "Cosine Similarity", "DTW Distance", "Spectrogram", "Fourier Transform",
                                                 command=self.update_motion_analysis_display)
        self.motion_analysis_menu.place(x=self.colored_frame.winfo_width()-310, y=10)
        
        # Create frame for motion analysis data
        self.motion_analysis_frame = tk.Frame(self.colored_frame, bg='black')
        self.motion_analysis_frame.place(x=self.colored_frame.winfo_width()-310, y=40, width=300, height=200)
        self.motion_analysis_frame.pack_forget()  # Hide initially
        
        # Create canvas for motion analysis visualization
        self.motion_analysis_canvas = tk.Canvas(self.motion_analysis_frame, bg='black', highlightthickness=0)
        self.motion_analysis_canvas.pack(fill='both', expand=True)
        
        # Initialize motion analysis data
        self.motion_data = {
            'cosine_similarity': [],
            'dtw_distance': [],
            'spectrogram': [],
            'fourier_transform': []
        }
        
        # Start update loop with reduced delay
        self.update()

    def on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """When the canvas is resized, resize the inner frame to match"""
        width = event.width
        self.control_canvas.itemconfig(self.control_canvas.find_withtag("all")[0], width=width)

    def connect_sensor(self):
        """Connect to the Phyphox sensor"""
        try:
            ip = self.ip_entry.get()
            port = int(self.port_entry.get())
            
            # Test connection
            url = f"http://{ip}:{port}/control?cmd=start"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                self.sensor_ip = ip
                self.sensor_port = port
                print(f"Connected to sensor at {ip}:{port}")
                # Update connection status to green
                self.connection_status.delete("all")
                self.connection_status.create_oval(0, 0, 20, 20, fill='green')
            else:
                print(f"Failed to connect to sensor: Status code {response.status_code}")
                # Update connection status to red
                self.connection_status.delete("all")
                self.connection_status.create_oval(0, 0, 20, 20, fill='red')
        except Exception as e:
            print(f"Error connecting to sensor: {e}")
            # Update connection status to red
            self.connection_status.delete("all")
            self.connection_status.create_oval(0, 0, 20, 20, fill='red')

    def update_sensor_data(self):
        """Update sensor data from Phyphox"""
        if not self.sensor_ip:
            self.connection_status.delete("all")
            self.connection_status.create_oval(0, 0, 20, 20, fill='red')
            return
            
        try:
            # Get gyroscope data
            gyro_url = f"http://{self.sensor_ip}:{self.sensor_port}/get?gyrX&gyrY&gyrZ&gyr_time"
            gyro_response = requests.get(gyro_url, timeout=1)
            
            # Get acceleration data
            accel_url = f"http://{self.sensor_ip}:{self.sensor_port}/get?accX&accY&accZ&acc_time"
            accel_response = requests.get(accel_url, timeout=1)
            
            if gyro_response.status_code == 200 and accel_response.status_code == 200:
                # Update connection status to green
                self.connection_status.delete("all")
                self.connection_status.create_oval(0, 0, 20, 20, fill='green')
                
                gyro_data = gyro_response.json()
                accel_data = accel_response.json()
                
                if 'buffer' in gyro_data and 'buffer' in accel_data:
                    gyro_buffer = gyro_data['buffer']
                    accel_buffer = accel_data['buffer']
                    
                    # Process gyroscope data
                    if all(key in gyro_buffer for key in ['gyrX', 'gyrY', 'gyrZ', 'gyr_time']):
                        self.process_gyro_data(gyro_buffer)
                    
                    # Process acceleration data
                    if all(key in accel_buffer for key in ['accX', 'accY', 'accZ', 'acc_time']):
                        self.process_accel_data(accel_buffer)
                    
                    # Update analysis and visualization
                    self.update_analysis()
                    self.update_sensor_visualization()
                    
        except Exception as e:
            print(f"Error updating sensor data: {e}")
            # Update connection status to red on error
            self.connection_status.delete("all")
            self.connection_status.create_oval(0, 0, 20, 20, fill='red')

    def process_gyro_data(self, buffer):
        """Process gyroscope data"""
        try:
            # Add new data points, filtering out None values
            self.sensor_analysis['gyro']['x'].extend([x for x in buffer['gyrX']['buffer'] if x is not None])
            self.sensor_analysis['gyro']['y'].extend([y for y in buffer['gyrY']['buffer'] if y is not None])
            self.sensor_analysis['gyro']['z'].extend([z for z in buffer['gyrZ']['buffer'] if z is not None])
            self.sensor_analysis['gyro']['time'].extend([t for t in buffer['gyr_time']['buffer'] if t is not None])
            
            # Calculate magnitude
            for i in range(len(self.sensor_analysis['gyro']['x'])):
                if all(x is not None for x in [self.sensor_analysis['gyro']['x'][i], 
                                             self.sensor_analysis['gyro']['y'][i], 
                                             self.sensor_analysis['gyro']['z'][i]]):
                    magnitude = np.sqrt(
                        self.sensor_analysis['gyro']['x'][i]**2 +
                        self.sensor_analysis['gyro']['y'][i]**2 +
                        self.sensor_analysis['gyro']['z'][i]**2
                    )
                    self.sensor_analysis['gyro']['magnitude'].append(magnitude)
            
            # Keep only last 1000 points
            max_points = 1000
            for key in self.sensor_analysis['gyro']:
                self.sensor_analysis['gyro'][key] = self.sensor_analysis['gyro'][key][-max_points:]
                
        except Exception as e:
            print(f"Error processing gyro data: {e}")

    def process_accel_data(self, buffer):
        """Process acceleration data"""
        try:
            # Add new data points, filtering out None values
            self.sensor_analysis['accel']['x'].extend([x for x in buffer['accX']['buffer'] if x is not None])
            self.sensor_analysis['accel']['y'].extend([y for y in buffer['accY']['buffer'] if y is not None])
            self.sensor_analysis['accel']['z'].extend([z for z in buffer['accZ']['buffer'] if z is not None])
            self.sensor_analysis['accel']['time'].extend([t for t in buffer['acc_time']['buffer'] if t is not None])
            
            # Calculate magnitude
            for i in range(len(self.sensor_analysis['accel']['x'])):
                if all(x is not None for x in [self.sensor_analysis['accel']['x'][i], 
                                             self.sensor_analysis['accel']['y'][i], 
                                             self.sensor_analysis['accel']['z'][i]]):
                    magnitude = np.sqrt(
                        self.sensor_analysis['accel']['x'][i]**2 +
                        self.sensor_analysis['accel']['y'][i]**2 +
                        self.sensor_analysis['accel']['z'][i]**2
                    )
                    self.sensor_analysis['accel']['magnitude'].append(magnitude)
            
            # Keep only last 1000 points
            max_points = 1000
            for key in self.sensor_analysis['accel']:
                self.sensor_analysis['accel'][key] = self.sensor_analysis['accel'][key][-max_points:]
                
        except Exception as e:
            print(f"Error processing accel data: {e}")

    def update_analysis(self):
        """Update the analysis display"""
        try:
            # Get the current sensor type
            sensor_type = self.sensor_type.get()
            data = self.sensor_analysis[sensor_type]
            
            if not data['time'] or not all(len(data[axis]) > 0 for axis in ['x', 'y', 'z']):
                return
                
            # Filter out None values and ensure arrays are the same length
            valid_indices = [i for i in range(len(data['time'])) 
                           if all(data[axis][i] is not None for axis in ['x', 'y', 'z', 'time'])]
            
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
                
                avg_x = np.mean(clean_data['x'])
                avg_y = np.mean(clean_data['y'])
                avg_z = np.mean(clean_data['z'])
                
                # Calculate total rotation (integrate angular velocity)
                dt = np.diff(clean_data['time'])
                # Use trapezoidal integration for better accuracy
                total_rotation_x = np.sum((np.array(clean_data['x'][:-1]) + np.array(clean_data['x'][1:])) * dt / 2)
                total_rotation_y = np.sum((np.array(clean_data['y'][:-1]) + np.array(clean_data['y'][1:])) * dt / 2)
                total_rotation_z = np.sum((np.array(clean_data['z'][:-1]) + np.array(clean_data['z'][1:])) * dt / 2)
                
                # Update analysis data
                self.sensor_analysis['max_gyro'] = max(max_x, max_y, max_z)
                self.sensor_analysis['avg_gyro'] = (avg_x + avg_y + avg_z) / 3
                self.sensor_analysis['total_rotation'] = {
                    'x': total_rotation_x,
                    'y': total_rotation_y,
                    'z': total_rotation_z
                }
                
            else:  # accelerometer
                # Calculate maximum and average for each axis
                max_x = max(abs(x) for x in clean_data['x'])
                max_y = max(abs(y) for y in clean_data['y'])
                max_z = max(abs(z) for z in clean_data['z'])
                
                avg_x = np.mean(clean_data['x'])
                avg_y = np.mean(clean_data['y'])
                avg_z = np.mean(clean_data['z'])
                
                # Calculate velocity (integrate acceleration)
                dt = np.diff(clean_data['time'])
                # Use trapezoidal integration for better accuracy
                velocity_x = np.cumsum((np.array(clean_data['x'][:-1]) + np.array(clean_data['x'][1:])) * dt / 2)
                velocity_y = np.cumsum((np.array(clean_data['y'][:-1]) + np.array(clean_data['y'][1:])) * dt / 2)
                velocity_z = np.cumsum((np.array(clean_data['z'][:-1]) + np.array(clean_data['z'][1:])) * dt / 2)
                
                # Calculate speed (magnitude of velocity) and apply scaling factor
                # Scale down the speed to be more realistic for arm movements
                speed_scaling_factor = 0.1  # Adjust this factor to get more realistic speeds
                speed = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2) * speed_scaling_factor
                
                # Update analysis data
                self.sensor_analysis['max_accel'] = max(max_x, max_y, max_z)
                self.sensor_analysis['avg_accel'] = (avg_x + avg_y + avg_z) / 3
                self.sensor_analysis['max_speed'] = max(speed) if len(speed) > 0 else 0
                self.sensor_analysis['avg_speed'] = np.mean(speed) if len(speed) > 0 else 0
            
            # Update analysis overlay
            self.update_analysis_overlay()
            
        except Exception as e:
            print(f"Error updating analysis: {e}")

    def update_analysis_overlay(self):
        """Update the analysis overlay display"""
        try:
            # Clear previous content
            self.analysis_overlay.delete("all")
            
            # Draw background
            self.analysis_overlay.create_rectangle(0, 0, 300, 200, fill='black', stipple='gray50')
            
            # Draw title with smaller font
            self.analysis_overlay.create_text(150, 15, text="Motion Analysis", fill='white', font=('Helvetica', 10, 'bold'))
            
            # Get current sensor type and data
            sensor_type = self.sensor_type.get()
            data = self.sensor_analysis[sensor_type]
            
            if not data['time']:
                return
            
            # Draw statistics based on sensor type with smaller font
            y_pos = 35
            if sensor_type == "gyro":
                stats = [
                    f"Max X: {max(abs(x) for x in data['x']):.1f} rad/s",
                    f"Max Y: {max(abs(y) for y in data['y']):.1f} rad/s",
                    f"Max Z: {max(abs(z) for z in data['z']):.1f} rad/s",
                    f"Total Rotation X: {self.sensor_analysis['total_rotation']['x']:.1f} rad",
                    f"Total Rotation Y: {self.sensor_analysis['total_rotation']['y']:.1f} rad",
                    f"Total Rotation Z: {self.sensor_analysis['total_rotation']['z']:.1f} rad"
                ]
            else:  # accelerometer
                stats = [
                    f"Max X: {max(abs(x) for x in data['x']):.1f} m/s²",
                    f"Max Y: {max(abs(y) for y in data['y']):.1f} m/s²",
                    f"Max Z: {max(abs(z) for z in data['z']):.1f} m/s²",
                    f"Max Speed: {self.sensor_analysis['max_speed']:.2f} m/s",
                    f"Avg Speed: {self.sensor_analysis['avg_speed']:.2f} m/s"
                ]
            
            for stat in stats:
                self.analysis_overlay.create_text(150, y_pos, text=stat, fill='white', font=('Helvetica', 8))
                y_pos += 20
            
            # Draw real-time indicators with smaller size
            if len(data['x']) > 0:
                # Current X value
                current_x = data['x'][-1]
                x_bar_width = min(200, (abs(current_x) / max(1, max(abs(x) for x in data['x']))) * 200)
                self.analysis_overlay.create_rectangle(50, 150, 50 + x_bar_width, 165, fill='red')
                self.analysis_overlay.create_text(250, 157, text=f"X: {current_x:.1f}", fill='white', font=('Helvetica', 8))
                
                # Current Y value
                current_y = data['y'][-1]
                y_bar_width = min(200, (abs(current_y) / max(1, max(abs(y) for y in data['y']))) * 200)
                self.analysis_overlay.create_rectangle(50, 170, 50 + y_bar_width, 185, fill='green')
                self.analysis_overlay.create_text(250, 177, text=f"Y: {current_y:.1f}", fill='white', font=('Helvetica', 8))
                
        except Exception as e:
            print(f"Error updating analysis overlay: {e}")

    def update_motion_analysis_overlay(self):
        """Update the motion analysis overlay with frequency spectrum and fourier transform"""
        try:
            # Skip if there's no frequency data
            if not hasattr(self.pose_estimator.pose_3d, 'frequency_data'):
                return
            
            # Create motion analysis overlay if it doesn't exist
            if not hasattr(self, 'motion_analysis_overlay'):
                self.motion_analysis_overlay = tk.Canvas(self.colored_frame, bg='black', highlightthickness=0)
                self.motion_analysis_overlay.place(x=10, y=10, width=300, height=200)
            
            # Clear previous content
            self.motion_analysis_overlay.delete("all")
            
            # Draw background
            self.motion_analysis_overlay.create_rectangle(0, 0, 300, 200, fill='black', stipple='gray50')
            
            # Draw title
            self.motion_analysis_overlay.create_text(150, 15, text="Motion Analysis", fill='white', font=('Helvetica', 12, 'bold'))
            
            # Get frequency data
            frequencies = self.pose_estimator.pose_3d.frequency_data['frequencies']
            magnitudes = self.pose_estimator.pose_3d.frequency_data['magnitudes']
            
            # Draw frequency spectrum (top half)
            width = 280
            height = 50
            margin = 40
            
            # Draw spectrum
            spectrum_y = margin + height
            bar_width = 2
            spacing = 1
            
            # Normalize magnitudes
            max_mag = max(magnitudes) if magnitudes and max(magnitudes) > 0 else 1
            for i in range(len(magnitudes)):
                x = margin + i * (bar_width + spacing)
                bar_height = (magnitudes[i] / max_mag) * height
                self.motion_analysis_overlay.create_rectangle(
                    x, spectrum_y - bar_height,
                    x + bar_width, spectrum_y,
                    fill='cyan'
                )
                
            # Draw labels
            self.motion_analysis_overlay.create_text(150, height + margin + 15, text="Frequency Spectrum", fill='white', font=('Helvetica', 8))
        
        except Exception as e:
            print(f"Error updating motion analysis overlay: {e}")

    def update_action_classification_display(self):
        """Add action classification display below motion analysis"""
        try:
            # Create action classification overlay if it doesn't exist
            if not hasattr(self, 'action_classification_overlay'):
                self.action_classification_overlay = tk.Canvas(self.colored_frame, bg='black', highlightthickness=0)
                # Position it properly below the motion analysis overlay
                self.action_classification_overlay.place(x=10, y=220, width=300, height=120)
            
            # Clear previous content
            self.action_classification_overlay.delete("all")
            
            # Draw background
            self.action_classification_overlay.create_rectangle(0, 0, 300, 120, fill='black', stipple='gray50')
            
            # Draw title with smaller font
            self.action_classification_overlay.create_text(150, 15, text="Sports Action Classification", fill='white', font=('Helvetica', 12, 'bold'))
            
            # Draw current action and percentage if available
            if hasattr(self, 'classification_result') and self.classification_result:
                action, confidence = self.classification_result
                
                # Draw action name with larger font
                self.action_classification_overlay.create_text(150, 45, 
                    text=f"{action}", 
                    fill='yellow', font=('Helvetica', 16, 'bold'))
                
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
                    fill='white', font=('Helvetica', 12))
            else:
                # No action detected
                self.action_classification_overlay.create_text(150, 60, 
                    text="No Action Detected", 
                    fill='white', font=('Helvetica', 14))
        except Exception as e:
            print(f"Error updating action classification display: {e}")
            print(f"Classification result: {self.classification_result if hasattr(self, 'classification_result') else 'None'}")

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
                
                # Run sports action classification if model is loaded
                if self.action_classifier is not None:
                    self.classify_action(frame)
                
        except Exception as e:
            print(f"Error in update: {str(e)}")
        
        # Schedule next update with reduced delay
        self.root.after(5, self.update)
    
    def classify_action(self, frame):
        """Classify sports action from the current frame"""
        if self.action_classifier is None:
            return
            
        try:
            # Use the action classifier to predict the action
            result = self.action_classifier.predict(frame)
            
            if result is not None and result[0] is not None:
                action, confidence = result
                
                # Store classification result but don't display
                self.classification_result = (action, confidence)
                # For debugging (reduce logging frequency)
                if hasattr(self, 'last_log_time'):
                    if time.time() - self.last_log_time > 2.0:  # Only log every 2 seconds
                        print(f"Detected action: {action} with confidence {confidence:.1f}%")
                        self.last_log_time = time.time()
                else:
                    self.last_log_time = time.time()
                    print(f"Detected action: {action} with confidence {confidence:.1f}%")
        except Exception as e:
            print(f"Error in sports action classification: {str(e)}")

    def run(self):
        """Start the main application loop"""
        self.root.mainloop()

    def quit_app(self):
        """Clean up and quit the application"""
        self.cap.release()
        pygame.quit()
        self.root.quit()

    def switch_camera(self):
        """Switch between available cameras"""
        self.current_camera = 1 if self.current_camera == 0 else 0
        self.cap.release()
        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            self.current_camera = 1 if self.current_camera == 0 else 0
            self.cap = cv2.VideoCapture(self.current_camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def toggle_background(self):
        """Toggle the background color of the 3D visualization"""
        self.pose_estimator.pose_3d.toggle_background()

    def reset_view(self):
        """Reset the 3D view to default position"""
        self.pose_estimator.pose_3d.camera_distance = 5.0
        self.pose_estimator.pose_3d.camera_x = 0.0
        self.pose_estimator.pose_3d.camera_y = 0.0
        self.pose_estimator.pose_3d.rotation_x = 0
        self.pose_estimator.pose_3d.rotation_y = 0

    def on_mouse_down(self, event):
        """Handle mouse button down events"""
        self.is_dragging = True
        self.last_x = event.x
        self.last_y = event.y
        self.colored_frame.focus_set()  # Ensure canvas has focus for events
        return "break"
    
    def on_mouse_up(self, event):
        """Handle mouse button up events"""
        self.is_dragging = False
        return "break"
    
    def on_mouse_drag(self, event):
        """Handle mouse drag events for 3D controls"""
        if not self.is_dragging:
            return
            
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        
        # Left mouse button for both rotation and panning
        # Use vertical movement for rotation
        self.pose_estimator.pose_3d.rotation_x += dy * 0.5
        # Use horizontal movement for panning
        self.pose_estimator.pose_3d.rotation_y += dx * 0.5
        
        self.last_x = event.x
        self.last_y = event.y
        return "break"
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming"""
        if event.delta > 0:
            self.pose_estimator.pose_3d.camera_distance = max(2, self.pose_estimator.pose_3d.camera_distance - 0.3)
        else:
            self.pose_estimator.pose_3d.camera_distance = min(10, self.pose_estimator.pose_3d.camera_distance + 0.3)
        return "break"

    def capture_ideal_pose(self):
        """Capture current pose as ideal pose"""
        try:
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_estimator.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    self.pose_estimator.pose_3d.ideal_pose = results.pose_landmarks.landmark
                    print("Ideal pose captured successfully!")
                else:
                    print("No pose detected. Please try again.")
        except Exception as e:
            print(f"Error capturing ideal pose: {str(e)}")

    def update_sensor_visualization(self):
        """Update the sensor data visualization"""
        try:
            # Clear previous visualization
            self.sensor_canvas.delete("all")
            
            # Redraw connection status and radio buttons
            self.connection_status.place(x=10, y=10)
            ttk.Radiobutton(self.sensor_canvas, text="Gyroscope", variable=self.sensor_type, value="gyro").place(x=40, y=10)
            ttk.Radiobutton(self.sensor_canvas, text="Accelerometer", variable=self.sensor_type, value="accel").place(x=150, y=10)
            
            # Get canvas dimensions
            width = self.sensor_canvas.winfo_width()
            height = self.sensor_canvas.winfo_height()
            
            if width <= 1 or height <= 1:
                return
            
            # Draw background
            self.sensor_canvas.create_rectangle(0, 0, width, height, fill='black')
            
            # Draw axes
            self.sensor_canvas.create_line(50, height-50, width-50, height-50, fill='gray', dash=(4, 4))  # X-axis
            self.sensor_canvas.create_line(50, height-50, 50, 50, fill='gray', dash=(4, 4))  # Y-axis
            
            # Get data based on selected sensor type
            if self.sensor_type.get() == "gyro":
                data = self.sensor_analysis['gyro']
                colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
                y_label = "Angular Velocity (rad/s)"
            else:
                data = self.sensor_analysis['accel']
                colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
                y_label = "Acceleration (m/s²)"
            
            # Plot data
            if data['time']:
                # Calculate scaling factors
                t_min, t_max = min(data['time']), max(data['time'])
                
                # Calculate data ranges for each axis
                d_ranges = {}
                for axis in ['x', 'y', 'z']:
                    d_min, d_max = min(data[axis]), max(data[axis])
                    d_ranges[axis] = (d_min, d_max)
                
                if t_max == t_min:
                    return
                
                # Plot each axis
                for axis, color in colors.items():
                    points = []
                    d_min, d_max = d_ranges[axis]
                    
                    # Add some padding to the data range
                    range_padding = (d_max - d_min) * 0.1
                    d_min -= range_padding
                    d_max += range_padding
                    
                    for i in range(len(data[axis])):
                        x = 50 + (data['time'][i] - t_min) / (t_max - t_min) * (width - 100)
                        y = height - 50 - (data[axis][i] - d_min) / (d_max - d_min) * (height - 100)
                        points.extend([x, y])
                    
                    if len(points) >= 4:
                        self.sensor_canvas.create_line(points, fill=color, smooth=True)
                
                # Add labels
                self.sensor_canvas.create_text(width/2, height-20, text="Time (s)", fill='white')
                self.sensor_canvas.create_text(30, height/2, text=y_label, fill='white', angle=90)
                
                # Add legend
                legend_y = 40
                for axis, color in colors.items():
                    self.sensor_canvas.create_line(width-100, legend_y, width-80, legend_y, fill=color)
                    self.sensor_canvas.create_text(width-70, legend_y, text=axis.upper(), fill='white', anchor='w')
                    legend_y += 20
                    
        except Exception as e:
            print(f"Error updating sensor visualization: {e}")

    def rotate_90_degrees(self):
        """Rotate the 3D view by 90 degrees"""
        self.pose_estimator.pose_3d.rotation_y += 90
        self.pose_estimator.pose_3d.render_pose(self.pose_estimator.pose_3d.current_pose)

    def update_motion_analysis_display(self, *args):
        """Update the motion analysis display based on selected option"""
        selected = self.motion_analysis_var.get()
        
        if selected == "Data":
            self.motion_analysis_frame.pack_forget()
        else:
            self.motion_analysis_frame.place(x=self.colored_frame.winfo_width()-310, y=40, width=300, height=200)
            self.update_motion_analysis_visualization(selected)

    def update_motion_analysis_visualization(self, analysis_type):
        """Update the motion analysis visualization"""
        try:
            self.motion_analysis_canvas.delete("all")
            
            # Draw background
            self.motion_analysis_canvas.create_rectangle(0, 0, 300, 200, fill='black', stipple='gray50')
            
            # Draw title
            self.motion_analysis_canvas.create_text(150, 15, text=analysis_type, fill='white', font=('Helvetica', 10, 'bold'))
            
            if analysis_type == "Cosine Similarity":
                if hasattr(self.pose_estimator.pose_3d, 'cosine_similarities') and self.pose_estimator.pose_3d.cosine_similarities:
                    similarities = self.pose_estimator.pose_3d.cosine_similarities
                    self.draw_line_graph(similarities, "Similarity", "red")
                    
            elif analysis_type == "DTW Distance":
                if hasattr(self.pose_estimator.pose_3d, 'dtw_distances') and self.pose_estimator.pose_3d.dtw_distances:
                    distances = self.pose_estimator.pose_3d.dtw_distances
                    self.draw_line_graph(distances, "Distance", "blue")
                    
            elif analysis_type == "Spectrogram":
                if hasattr(self.pose_estimator.pose_3d, 'frequency_data'):
                    magnitudes = self.pose_estimator.pose_3d.frequency_data['magnitudes']
                    self.draw_spectrogram(magnitudes)
                    
            elif analysis_type == "Fourier Transform":
                if hasattr(self.pose_estimator.pose_3d, 'frequency_data'):
                    frequencies = self.pose_estimator.pose_3d.frequency_data['frequencies']
                    magnitudes = self.pose_estimator.pose_3d.frequency_data['magnitudes']
                    self.draw_fourier_transform(frequencies, magnitudes)
                    
        except Exception as e:
            print(f"Error updating motion analysis visualization: {e}")

    def draw_line_graph(self, data, label, color):
        """Draw a line graph of the data"""
        if not data:
            return
            
        width = 280
        height = 150
        margin = 20
        
        # Calculate scaling factors
        x_scale = width / (len(data) - 1) if len(data) > 1 else 1
        y_scale = height / (max(data) - min(data)) if max(data) != min(data) else 1
        
        # Draw axes
        self.motion_analysis_canvas.create_line(margin, margin, margin, height + margin, fill='gray')
        self.motion_analysis_canvas.create_line(margin, height + margin, width + margin, height + margin, fill='gray')
        
        # Draw data line
        points = []
        for i, value in enumerate(data):
            x = margin + i * x_scale
            y = height + margin - (value - min(data)) * y_scale
            points.extend([x, y])
            
        if len(points) >= 4:
            self.motion_analysis_canvas.create_line(points, fill=color, smooth=True)
            
        # Draw labels
        self.motion_analysis_canvas.create_text(150, height + margin + 15, text=label, fill='white', font=('Helvetica', 8))

    def draw_spectrogram(self, magnitudes):
        """Draw a spectrogram visualization"""
        if not magnitudes:
            return
            
        width = 280
        height = 150
        margin = 20
        
        # Normalize magnitudes
        max_mag = max(magnitudes)
        if max_mag == 0:
            return
            
        # Draw spectrogram
        bar_width = 2
        spacing = 1
        num_bars = min(len(magnitudes), width // (bar_width + spacing))
        
        for i in range(num_bars):
            x = margin + i * (bar_width + spacing)
            height_value = (magnitudes[i] / max_mag) * height
            self.motion_analysis_canvas.create_rectangle(
                x, height + margin - height_value,
                x + bar_width, height + margin,
                fill='blue'
            )
            
        # Draw labels
        self.motion_analysis_canvas.create_text(150, height + margin + 15, text="Frequency Spectrum", fill='white', font=('Helvetica', 8))

    def draw_fourier_transform(self, frequencies, magnitudes):
        """Draw a Fourier transform visualization"""
        if not frequencies or not magnitudes:
            return
            
        width = 280
        height = 150
        margin = 20
        
        # Normalize magnitudes
        max_mag = max(magnitudes)
        if max_mag == 0:
            return
            
        # Draw frequency components
        points = []
        for i in range(len(frequencies)):
            x = margin + (frequencies[i] - min(frequencies)) / (max(frequencies) - min(frequencies)) * width
            y = height + margin - (magnitudes[i] / max_mag) * height
            points.extend([x, y])
            
        if len(points) >= 4:
            self.motion_analysis_canvas.create_line(points, fill='green', smooth=True)
            
        # Draw labels
        self.motion_analysis_canvas.create_text(150, height + margin + 15, text="Fourier Transform", fill='white', font=('Helvetica', 8))

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
        """Process a frame and make a prediction when conditions are met."""
        if not self.model_ready:
            return None, 0
        
        # Add frame to buffer
        self.add_frame(frame)
        
        # Return if buffer not full
        if not self.is_buffer_full():
            return None, 0
        
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
            return None, 0

def main():
    app = PoseVisualizerGUI()
    app.run()

if __name__ == "__main__":
    main() 