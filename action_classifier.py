import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, TimeDistributed, ConvLSTM2D
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

# Action recognition label map
ACTION_LABELS = {
    "PullUps": 0,
    "PushUps": 1, 
    "JugglingBalls": 2,
    "BoxingSpeedBag": 3,
    "Punch": 4
}

# Simple Action Recognition Model
class SimpleActionRecognitionModel:
    def __init__(self):
        self.label_map = ACTION_LABELS
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.build_model()
        
    def build_model(self):
        """Build a simple action recognition model"""
        # Input shape: (batch_size, sequence_length, height, width, channels)
        input_shape = (30, 60, 60, 3)
        
        model = Sequential([
            # First ConvLSTM layer
            ConvLSTM2D(32, (3, 3), activation='relu', 
                      return_sequences=True, 
                      input_shape=input_shape),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Dropout(0.25)),
            
            # Second ConvLSTM layer
            ConvLSTM2D(64, (3, 3), activation='relu', 
                      return_sequences=True),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Dropout(0.25)),
            
            # Third ConvLSTM layer
            ConvLSTM2D(128, (3, 3), activation='relu', 
                      return_sequences=False),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(self.label_map), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Simple action recognition model created")
        
    def predict(self, frames_array, verbose=0):
        """Make a prediction on a series of frames"""
        # Ensure correct input shape
        if frames_array.shape[1:] != (30, 60, 60, 3):
            frames_array = tf.image.resize(frames_array, (60, 60))
            frames_array = np.reshape(frames_array, (1, 30, 60, 60, 3))
            
        # Make prediction
        return self.model.predict(frames_array, verbose=verbose)
        
    def random_predict(self):
        """Generate a random prediction for demonstration purposes"""
        # Create a random probability distribution across classes
        probs = np.random.random(len(self.label_map))
        probs = probs / np.sum(probs)  # Normalize to sum to 1
        
        # Return as a batch prediction (shape: [1, num_classes])
        return np.array([probs])

class ActionClassifierApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Action Classifier")
        self.root.geometry("800x600")
        
        # Initialize video capture with fallback
        self.current_camera = 0
        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            self.current_camera = 1 if self.current_camera == 0 else 0
            self.cap = cv2.VideoCapture(self.current_camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize action recognition model
        self.action_model = SimpleActionRecognitionModel()
        self.label_map = self.action_model.label_map
        self.reverse_label_map = self.action_model.reverse_label_map
        
        # Action recognition variables
        self.action_frames = []
        self.max_frames = 30  # Number of frames to collect before classification
        self.classification_result = None
        self.last_classification_time = 0
        self.classification_interval = 1.0  # Classify every 1 second
        self.is_exercise_active = False
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Video frame
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(pady=10)
        
        # Controls frame
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.pack(fill='x', pady=10)
        
        # Exercise dropdown
        self.exercise_var = tk.StringVar()
        ttk.Label(controls_frame, text="Select Exercise:").pack(side='left', padx=5)
        self.exercise_dropdown = ttk.Combobox(controls_frame, 
                                            textvariable=self.exercise_var,
                                            values=list(self.label_map.keys()),
                                            state='readonly')
        self.exercise_dropdown.pack(side='left', padx=5)
        self.exercise_dropdown.current(0)  # Select first exercise by default
        
        # Start/Stop button
        self.button_text = tk.StringVar()
        self.button_text.set("Start Exercise")
        self.start_button = ttk.Button(controls_frame, 
                                     textvariable=self.button_text,
                                     command=self.toggle_exercise)
        self.start_button.pack(side='left', padx=5)
        
        # Classification display frame
        self.class_frame = ttk.LabelFrame(self.main_frame, text="Classification Result")
        self.class_frame.pack(fill='x', pady=10, padx=20)
        
        # Action label
        self.action_label = ttk.Label(self.class_frame, text="No Action Detected", font=('Helvetica', 16, 'bold'))
        self.action_label.pack(pady=5)
        
        # Confidence bar frame
        confidence_frame = ttk.Frame(self.class_frame)
        confidence_frame.pack(fill='x', pady=5)
        
        # Confidence bar
        self.confidence_bar = ttk.Progressbar(confidence_frame, length=700, mode='determinate')
        self.confidence_bar.pack(side='left', padx=5)
        
        # Confidence label
        self.confidence_label = ttk.Label(confidence_frame, text="0%")
        self.confidence_label.pack(side='left', padx=5)
        
        # Start update loop
        self.update()
        
        # Run the app
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def toggle_exercise(self):
        """Toggle exercise mode on/off"""
        self.is_exercise_active = not self.is_exercise_active
        if self.is_exercise_active:
            self.button_text.set("Stop Exercise")
            self.action_frames = []  # Reset action frames
        else:
            self.button_text.set("Start Exercise")
            self.action_frames = []  # Reset action frames
            self.action_label.config(text="No Action Detected")
            self.confidence_bar['value'] = 0
            self.confidence_label.config(text="0%")
            
    def update(self):
        """Update the app"""
        ret, frame = self.cap.read()
        if ret:
            # Resize frame for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Convert to RGB and then to PhotoImage
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update video label
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk
            
            # If exercise active, classify the action
            if self.is_exercise_active:
                self.classify_action(frame)
        
        # Schedule next update
        self.root.after(33, self.update)  # ~30 FPS
        
    def classify_action(self, frame):
        """Classify action from the current frame"""
        current_time = time.time()
        if current_time - self.last_classification_time < self.classification_interval:
            return
            
        try:
            # Preprocess frame for classification
            frame = cv2.resize(frame, (60, 60))  # Adjusted size to match model input
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)
            
            # Add frame to buffer
            self.action_frames.append(frame)
            
            # Keep only last max_frames frames
            if len(self.action_frames) > self.max_frames:
                self.action_frames.pop(0)
                
            # If we have enough frames, perform classification
            if len(self.action_frames) == self.max_frames:
                # Use random predictions for testing
                prediction = self.action_model.random_predict()
                
                action_idx = np.argmax(prediction[0])
                confidence = prediction[0][action_idx] * 100
                
                # Get the predicted action name
                predicted_action = self.reverse_label_map[action_idx]
                
                # Update UI
                self.action_label.config(text=predicted_action)
                self.confidence_bar['value'] = confidence
                self.confidence_label.config(text=f"{confidence:.1f}%")
                
                # Check if it matches selected exercise
                selected_exercise = self.exercise_var.get()
                if predicted_action == selected_exercise:
                    self.action_label.config(foreground='green')
                else:
                    self.action_label.config(foreground='red')
                
                self.last_classification_time = current_time
        except Exception as e:
            print(f"Error in action classification: {str(e)}")
            
    def on_closing(self):
        """Called when the window is closing"""
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = ActionClassifierApp() 