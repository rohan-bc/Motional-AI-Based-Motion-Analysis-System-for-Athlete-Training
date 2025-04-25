"""
AI-Powered Sports Feedback System

This module provides real-time coaching and feedback for 10 different sports activities
by analyzing pose estimation data from MediaPipe, motion sensor data from PhyPhox,
and AI action recognition.

It uses DeepSeek AI via OpenRouter for generating high-quality coaching feedback.
"""

import os
import json
import time
import numpy as np
import requests
from collections import deque
import tkinter as tk

# OpenRouter API Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3-base:free"

# List of supported sports/exercises
SUPPORTED_EXERCISES = [
    "TennisSwing", "BoxingPunchingBag", "Diving", "Archery", 
    "Basketball", "LongJump", "JugglingBalls", "PushUps", 
    "BreastStroke", "PullUps"
]

# Form rules for each exercise (used for rule-based feedback)
FORM_RULES = {
    "TennisSwing": {
        "shoulder_hip_separation": 30,  # Minimum degrees
        "elbow_lock_threshold": 160,    # Degrees (>160 is locked)
        "follow_through_height": 0.7    # Shoulder height ratio
    },
    "BoxingPunchingBag": {
        "guard_drop_threshold": 0.2,    # Maximum drop ratio
        "hip_rotation_min": 15,         # Minimum degrees
        "recovery_delay_max": 1.0       # Maximum seconds
    },
    "Diving": {
        "knee_bend_max": 10,            # Maximum degrees in flight
        "tuck_tightness": 45,           # Maximum degrees for tight tuck
        "takeoff_force_min": 5.0        # m/s² minimum
    },
    "Archery": {
        "min_draw_time": 2.0,           # Seconds
        "bow_arm_stability": 0.02,      # Maximum movement
        "torso_lean_max": 5             # Maximum degrees
    },
    "Basketball": {
        "wrist_flick_min": 30,          # Minimum degrees
        "vertical_drift_max": 0.1,      # Maximum ratio
        "elbow_alignment_max": 10       # Maximum degrees off alignment
    },
    "LongJump": {
        "takeoff_angle_min": 15,        # Minimum degrees
        "arm_swing_asymmetry_max": 10,  # Maximum percent difference
        "knee_buckle_threshold": 20     # Maximum degrees
    },
    "JugglingBalls": {
        "toss_height_min": 0.3,         # Minimum ratio
        "head_movement_max": 0.05,      # Maximum meters
        "catch_force_max": 2.0          # Maximum m/s²
    },
    "PushUps": {
        "chest_height_max": 0.1,        # Maximum meters from ground
        "hip_sag_max": 10,              # Maximum degrees
        "tempo_min": 1.0                # Minimum seconds per rep
    },
    "BreastStroke": {
        "ankle_spread_min": 0.3,        # Minimum meters
        "glide_time_min": 0.5,          # Minimum seconds
        "head_lift_max": 0.05           # Maximum meters
    },
    "PullUps": {
        "extension_min": 160,           # Minimum degrees for full extension
        "max_swing": 0.1,               # Maximum gyro reading
        "shoulder_shrug_max": 10        # Maximum degrees
    }
}

class GeminiCoach:
    def __init__(self):
        """Initialize the AI coach with required parameters and history."""
        self.feedback_history = deque(maxlen=5)
        self.sensor_history = {
            'gyro': deque(maxlen=100),
            'accel': deque(maxlen=100)
        }
        self.landmark_history = deque(maxlen=30)
        self.rep_count = {exercise: 0 for exercise in SUPPORTED_EXERCISES}
        self.form_scores = {exercise: [] for exercise in SUPPORTED_EXERCISES}
        self.current_exercise = None
        self.exercise_active = False
        self.start_time = None
        self.last_feedback_time = 0
        self.feedback_interval = 3.0  # Seconds between feedback
        
        # Performance metrics
        self.metrics = {
            'rep_count': 0,
            'form_score': 0,
            'tempo': 0,
            'fatigue_level': 0,
            'range_of_motion': 0
        }
        
        # Session stats
        self.session_data = {
            'duration': 0,
            'reps_by_exercise': {},
            'avg_form_score': 0,
            'max_form_score': 0,
            'improvement_tips': []
        }
        
        # Error counter to keep track of common form issues
        self.error_counter = {}
        
        print("GeminiCoach initialized successfully")
    
    def start_exercise_session(self, exercise_name):
        """Start a new exercise session."""
        if exercise_name not in SUPPORTED_EXERCISES:
            print(f"Warning: {exercise_name} is not a supported exercise. Defaulting to general feedback.")
        
        self.current_exercise = exercise_name
        self.exercise_active = True
        self.start_time = time.time()
        self.rep_count[exercise_name] = 0
        self.form_scores[exercise_name] = []
        self.error_counter = {}
        
        print(f"Started {exercise_name} session")
        
        # Return initial motivation
        return f"Starting {exercise_name} session. Get ready!"
    
    def end_exercise_session(self):
        """End the current exercise session and return summary."""
        if not self.exercise_active:
            return "No active exercise session to end."
        
        duration = time.time() - self.start_time
        avg_form = np.mean(self.form_scores[self.current_exercise]) if self.form_scores[self.current_exercise] else 0
        
        self.session_data['duration'] = duration
        self.session_data['reps_by_exercise'][self.current_exercise] = self.rep_count[self.current_exercise]
        self.session_data['avg_form_score'] = avg_form
        self.session_data['max_form_score'] = max(self.form_scores[self.current_exercise]) if self.form_scores[self.current_exercise] else 0
        
        # Generate improvement tips based on error counter
        tips = self._generate_improvement_tips()
        self.session_data['improvement_tips'] = tips
        
        self.exercise_active = False
        print(f"Ended {self.current_exercise} session")
        
        return {
            'summary': self._generate_session_summary(),
            'tips': tips
        }
    
    def add_sensor_data(self, sensor_type, data):
        """Add sensor data to history for analysis."""
        if sensor_type in ['gyro', 'accel']:
            self.sensor_history[sensor_type].append(data)
    
    def add_landmark_data(self, landmarks):
        """Add pose landmark data to history for analysis."""
        self.landmark_history.append(landmarks)
    
    def get_feedback(self, action, landmarks, sensor_data, confidence):
        """Generate AI coaching feedback based on current data."""
        current_time = time.time()
        
        # Only provide feedback at the specified interval
        if current_time - self.last_feedback_time < self.feedback_interval:
            return None
        
        self.last_feedback_time = current_time
        
        # Check if we have a valid exercise and action classification
        if not self.exercise_active or not self.current_exercise:
            return "Start an exercise session to get feedback."
        
        # Check if the detected action matches the current exercise
        action_match = action == self.current_exercise
        
        # Use rule-based feedback first
        rule_feedback = self._apply_form_rules(landmarks, sensor_data)
        
        # If we have rule-based feedback, use that
        if rule_feedback:
            self.feedback_history.append(rule_feedback)
            return rule_feedback
        
        # If no rule-based feedback, use Gemini API for more nuanced feedback
        return self._get_ai_feedback(action, landmarks, sensor_data, confidence, action_match)
    
    def _apply_form_rules(self, landmarks, sensor_data):
        """Apply rule-based form checking for the current exercise."""
        if not self.current_exercise or self.current_exercise not in FORM_RULES:
            return None
        
        rules = FORM_RULES[self.current_exercise]
        feedback = None
        
        # Extract relevant joint angles and positions
        joint_data = self._extract_joint_data(landmarks)
        
        # Apply exercise-specific rules
        if self.current_exercise == "PushUps":
            # Check for hip sagging
            if joint_data.get('hip_angle') and joint_data['hip_angle'] < 160:
                feedback = "Don't sag your hips! Keep your body in a straight line."
                self._count_error("hip_sag")
            
            # Check for chest height (depth)
            if joint_data.get('elbow_angle') and joint_data['elbow_angle'] > 90:
                feedback = "Go lower! Your chest should nearly touch the ground."
                self._count_error("shallow_pushup")
        
        elif self.current_exercise == "PullUps":
            # Check for full extension
            if joint_data.get('elbow_angle') and joint_data['elbow_angle'] < rules['extension_min']:
                feedback = "Extend your arms fully at the bottom of each rep!"
                self._count_error("incomplete_extension")
            
            # Check for excessive swinging
            if sensor_data and sensor_data.get('gyro'):
                gyro_mag = np.sqrt(sensor_data['gyro']['x']**2 + 
                                 sensor_data['gyro']['y']**2 + 
                                 sensor_data['gyro']['z']**2)
                if gyro_mag > rules['max_swing']:
                    feedback = "Stop swinging! Keep your body controlled."
                    self._count_error("excessive_swing")
        
        # Add more exercise-specific rules here
        
        return feedback
    
    def _get_ai_feedback(self, action, landmarks, sensor_data, confidence, action_match):
        """Generate AI coaching feedback using API or fallback to simulated feedback if API fails."""
        try:
            # Prepare the prompt for the API
            prompt = self._create_feedback_prompt(action, landmarks, sensor_data, confidence, action_match)
            
            # Call the API
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert sports coach providing concise, helpful feedback on exercise form and technique."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150
            }
            
            try:
                response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=3)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Debug: Print the actual response structure
                    print(f"API Response Structure: {json.dumps(response_data, indent=2)}")
                    
                    # Try different response formats
                    feedback = None
                    
                    # Format 1: OpenRouter standard format with choices
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        if 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                            feedback = response_data['choices'][0]['message']['content'].strip()
                    
                    # Format 2: Direct Gemini format with candidates
                    elif 'candidates' in response_data and len(response_data['candidates']) > 0:
                        if 'content' in response_data['candidates'][0]:
                            if 'parts' in response_data['candidates'][0]['content']:
                                parts = response_data['candidates'][0]['content']['parts']
                                if parts and 'text' in parts[0]:
                                    feedback = parts[0]['text'].strip()
                    
                    # Format 3: Alternative format with generations
                    elif 'generations' in response_data and len(response_data['generations']) > 0:
                        if 'text' in response_data['generations'][0]:
                            feedback = response_data['generations'][0]['text'].strip()
                    
                    # If no known format matched, try a simple fallback approach
                    if feedback is None:
                        # Walk through the response to find text content
                        def find_text_content(obj):
                            if isinstance(obj, str):
                                return obj
                            elif isinstance(obj, dict):
                                for key in ['content', 'text', 'message']:
                                    if key in obj:
                                        result = find_text_content(obj[key])
                                        if result:
                                            return result
                                for value in obj.values():
                                    result = find_text_content(value)
                                    if result:
                                        return result
                            elif isinstance(obj, list):
                                for item in obj:
                                    result = find_text_content(item)
                                    if result:
                                        return result
                            return None
                        
                        feedback = find_text_content(response_data)
                    
                    if feedback:
                        self.feedback_history.append(feedback)
                        return feedback
                    else:
                        print(f"Could not extract feedback from API response, using simulated feedback")
                        return self._generate_simulated_feedback(action)
                elif response.status_code == 429:
                    print(f"API rate limit exceeded, using simulated feedback")
                    return self._generate_simulated_feedback(action)
                else:
                    print(f"API Error: {response.status_code} - {response.text}, using simulated feedback")
                    return self._generate_simulated_feedback(action)
            except requests.exceptions.RequestException as e:
                print(f"API Request error: {e}, using simulated feedback")
                return self._generate_simulated_feedback(action)
        
        except Exception as e:
            print(f"Error getting AI feedback: {e}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return self._generate_simulated_feedback(action)
    
    def _create_feedback_prompt(self, action, landmarks, sensor_data, confidence, action_match):
        """Create a prompt for the Gemini API based on current exercise data."""
        # Create a condensed representation of landmarks and sensor data
        landmark_summary = self._summarize_landmarks(landmarks)
        sensor_summary = self._summarize_sensor_data(sensor_data)
        
        # Add analysis data to prompt if available
        analysis_summary = ""
        if hasattr(self, 'analysis_data') and self.analysis_data:
            analysis_summary = "Motion Analysis Data:\n"
            for data in self.analysis_data[-3:]:  # Include last 3 analysis points
                if data['type'] == 'cosine_similarity':
                    analysis_summary += f"Movement consistency: {data['mean']:.2f} (0-1 scale, higher is better)\n"
                elif data['type'] == 'dtw':
                    analysis_summary += f"Form deviation: {data['mean']:.2f} (lower is better)\n"
                elif data['type'] == 'fourier':
                    analysis_summary += f"Movement rhythm: {data['rhythm_consistency']:.2f} at {data['dominant_frequency']:.3f}Hz\n"
        
        # Build the prompt
        prompt = f"""
        As an expert sports coach, provide a single sentence of specific, actionable feedback for a person doing {self.current_exercise}.
        
        Exercise: {self.current_exercise}
        Detected Action: {action} (Confidence: {confidence:.1f}%)
        Correct Action Match: {"Yes" if action_match else "No"}
        
        Body Position:
        {landmark_summary}
        
        Sensor Data:
        {sensor_summary}
        
        {analysis_summary}
        
        Reps Completed: {self.rep_count[self.current_exercise]}
        
        Common Errors:
        {json.dumps(self.error_counter, indent=2)}
        
        Previous Feedback:
        {", ".join(list(self.feedback_history)[-2:]) if len(self.feedback_history) >= 2 else "None"}
        
        Provide specific, concise coaching feedback (max 15 words) to improve form or encourage the athlete.
        """
        
        return prompt
    
    def _summarize_landmarks(self, landmarks):
        """Create a summary of key landmark positions for the prompt."""
        if not landmarks:
            return "No landmark data available"
        
        # Extract key joint angles
        joint_data = self._extract_joint_data(landmarks)
        
        summary = []
        for joint, value in joint_data.items():
            if isinstance(value, (int, float)):
                summary.append(f"{joint}: {value:.1f}")
            else:
                summary.append(f"{joint}: {value}")
        
        return "\n".join(summary)
    
    def _summarize_sensor_data(self, sensor_data):
        """Create a summary of sensor data for the prompt."""
        if not sensor_data:
            return "No sensor data available"
        
        summary = []
        
        # Process gyroscope data
        if 'gyro' in sensor_data:
            gyro = sensor_data['gyro']
            gyro_mag = np.sqrt(gyro['x']**2 + gyro['y']**2 + gyro['z']**2) if all(k in gyro for k in ['x', 'y', 'z']) else 0
            summary.append(f"Gyroscope: x={gyro.get('x', 0):.2f}, y={gyro.get('y', 0):.2f}, z={gyro.get('z', 0):.2f}, magnitude={gyro_mag:.2f}")
        
        # Process accelerometer data
        if 'accel' in sensor_data:
            accel = sensor_data['accel']
            accel_mag = np.sqrt(accel['x']**2 + accel['y']**2 + accel['z']**2) if all(k in accel for k in ['x', 'y', 'z']) else 0
            summary.append(f"Accelerometer: x={accel.get('x', 0):.2f}, y={accel.get('y', 0):.2f}, z={accel.get('z', 0):.2f}, magnitude={accel_mag:.2f}")
        
        return "\n".join(summary)
    
    def _extract_joint_data(self, landmarks):
        """Extract joint angles and positions from landmarks."""
        if not landmarks:
            return {}
        
        joint_data = {}
        
        # Calculate relevant joint angles based on landmarks
        # This is a simplified version - a real implementation would have more detailed calculations
        
        # Example: Calculate elbow angle (if landmarks have the right format)
        try:
            if hasattr(landmarks[0], 'x'):  # MediaPipe format
                # Extract shoulder, elbow, wrist landmarks
                shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15  # Right arm indices
                
                shoulder = landmarks[shoulder_idx]
                elbow = landmarks[elbow_idx]
                wrist = landmarks[wrist_idx]
                
                # Calculate vectors
                upper_arm = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z])
                forearm = np.array([wrist.x - elbow.x, wrist.y - elbow.y, wrist.z - elbow.z])
                
                # Normalize vectors
                upper_arm = upper_arm / np.linalg.norm(upper_arm)
                forearm = forearm / np.linalg.norm(forearm)
                
                # Calculate dot product and angle
                dot_product = np.dot(upper_arm, forearm)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
                
                # Convert to the angle we want (180 is straight arm)
                elbow_angle = 180 - angle
                
                joint_data['elbow_angle'] = elbow_angle
                
                # Similarly calculate other joint angles as needed
                # ...
            
        except Exception as e:
            print(f"Error extracting joint data: {e}")
        
        return joint_data
    
    def _count_error(self, error_type):
        """Count occurrences of specific errors for feedback prioritization."""
        if error_type not in self.error_counter:
            self.error_counter[error_type] = 1
        else:
            self.error_counter[error_type] += 1
    
    def _generate_improvement_tips(self):
        """Generate improvement tips based on the error counter."""
        tips = []
        
        # Sort errors by frequency
        sorted_errors = sorted(self.error_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Generate tips for the top 3 errors
        for error, count in sorted_errors[:3]:
            if error == "hip_sag":
                tips.append("Practice plank holds to strengthen your core and prevent hip sagging.")
            elif error == "shallow_pushup":
                tips.append("Focus on full range of motion - use a mirror or camera to check depth.")
            elif error == "incomplete_extension":
                tips.append("Practice dead hangs to get comfortable with full arm extension.")
            elif error == "excessive_swing":
                tips.append("Strengthen your core and focus on controlled movements to prevent swinging.")
            # Add more tips for other error types
        
        # If no specific errors, give general tips
        if not tips:
            tips.append("Great job! Keep practicing consistently to improve your technique.")
            
            if self.current_exercise == "PushUps":
                tips.append("Try different hand positions to work different muscle groups.")
            elif self.current_exercise == "PullUps":
                tips.append("Incorporate negative pullups to build strength for full pullups.")
            # Add general tips for other exercises
        
        return tips
    
    def _generate_session_summary(self):
        """Generate a summary of the exercise session."""
        duration_mins = (self.session_data['duration'] / 60) if 'duration' in self.session_data else 0
        reps = self.rep_count[self.current_exercise] if self.current_exercise else 0
        avg_form = np.mean(self.form_scores[self.current_exercise]) if self.current_exercise and self.form_scores.get(self.current_exercise) else 0
        
        summary = f"Session Summary: {self.current_exercise}\n"
        summary += f"Duration: {duration_mins:.1f} minutes\n"
        summary += f"Reps Completed: {reps}\n"
        summary += f"Average Form Score: {avg_form:.1f}%\n"
        
        # Add most common errors
        if self.error_counter:
            sorted_errors = sorted(self.error_counter.items(), key=lambda x: x[1], reverse=True)
            summary += "Common issues:\n"
            for error, count in sorted_errors[:3]:
                summary += f"- {error.replace('_', ' ').title()}: {count} times\n"
        
        return summary

    def count_rep(self, landmarks, action):
        """Count a repetition based on landmark positions and current action."""
        if not self.exercise_active or action != self.current_exercise:
            return False
        
        # Extract joint data
        joint_data = self._extract_joint_data(landmarks)
        
        # Implement rep counting logic based on the exercise
        if self.current_exercise == "PushUps":
            # Count a pushup when chest is near the ground (elbows at 90°) and then back up
            if joint_data.get('elbow_angle'):
                # Simplified rep counting
                if joint_data['elbow_angle'] < 90:  # Near bottom position
                    if not hasattr(self, 'pushup_bottom'):
                        self.pushup_bottom = True
                elif joint_data['elbow_angle'] > 160:  # Near top position
                    if hasattr(self, 'pushup_bottom') and self.pushup_bottom:
                        self.pushup_bottom = False
                        self.rep_count[self.current_exercise] += 1
                        return True
        
        # Add rep counting logic for other exercises
        
        return False

    def calculate_form_score(self, landmarks, action):
        """Calculate a form score (0-100) based on landmark positions and current action."""
        if not self.exercise_active or action != self.current_exercise:
            return 0
        
        joint_data = self._extract_joint_data(landmarks)
        form_score = 0
        
        # Implement form scoring logic based on the exercise
        if self.current_exercise == "PushUps":
            # Score based on body alignment and elbow angle
            # Start with perfect score and deduct for form errors
            form_score = 100
            
            # Check hip alignment
            if joint_data.get('hip_angle') and joint_data['hip_angle'] < 160:
                # Deduct points for hip sagging
                hip_penalty = (160 - joint_data['hip_angle']) * 1.5
                form_score -= min(40, hip_penalty)  # Cap the penalty
            
            # Check elbow angle at bottom
            if joint_data.get('elbow_angle') and joint_data['elbow_angle'] < 90:
                # Good depth, no penalty
                pass
            elif joint_data.get('elbow_angle') and joint_data['elbow_angle'] < 120:
                # Not deep enough, deduct points
                depth_penalty = (joint_data['elbow_angle'] - 90) * 1.0
                form_score -= min(30, depth_penalty)
            
            # Add more criteria here
        
        # Add form scoring logic for other exercises
        
        # Ensure score is between 0-100
        form_score = max(0, min(100, form_score))
        
        # Store form score for this exercise
        self.form_scores[self.current_exercise].append(form_score)
        
        return form_score

    def add_analysis_data(self, analysis_data):
        """Add analysis data to the coach for enhanced feedback"""
        try:
            if not hasattr(self, 'analysis_data'):
                self.analysis_data = []
            
            # Add timestamp to analysis data
            analysis_data['timestamp'] = time.time()
            self.analysis_data.append(analysis_data)
            
            # Keep only recent analysis data
            if len(self.analysis_data) > 10:
                self.analysis_data.pop(0)
                
            print(f"Added {analysis_data['type']} analysis data to AI coach")
            
        except Exception as e:
            print(f"Error adding analysis data to AI coach: {e}")

    def _generate_simulated_feedback(self, action):
        """Generate simulated coaching feedback when API is unavailable."""
        # Dictionary of feedback options for each exercise
        feedback_options = {
            "TennisSwing": [
                "Rotate your shoulders and hips for more power.",
                "Follow through completely with your swing.",
                "Keep your eye on the ball throughout your swing.",
                "Maintain a balanced stance during your swing."
            ],
            "BoxingPunchingBag": [
                "Keep your guard up between punches.",
                "Rotate your hips to add power to your punches.",
                "Stay light on your feet and maintain your stance.",
                "Exhale sharply with each punch for better power."
            ],
            "Diving": [
                "Keep your body streamlined during entry.",
                "Maintain a tight tuck during rotation.",
                "Extend fully before entering the water.",
                "Push off with more power from the platform."
            ],
            "Archery": [
                "Keep your bow arm steady throughout the draw.",
                "Anchor consistently at the same point each time.",
                "Release smoothly without jerking the string.",
                "Maintain your form after the release."
            ],
            "Basketball": [
                "Follow through with your wrist on each shot.",
                "Keep your elbow tucked in for better accuracy.",
                "Bend your knees for more power in your shot.",
                "Focus on a spot on the rim when shooting."
            ],
            "LongJump": [
                "Drive your knees higher during takeoff.",
                "Extend your arms forward during landing.",
                "Maintain speed through the whole approach.",
                "Focus on explosive takeoff from the board."
            ],
            "JugglingBalls": [
                "Keep a consistent rhythm with your throws.",
                "Throw to a consistent height for better control.",
                "Keep your hands at a steady height.",
                "Focus on the pattern, not individual balls."
            ],
            "PushUps": [
                "Keep your core tight throughout the movement.",
                "Lower your chest closer to the ground.",
                "Maintain a straight line from head to heels.",
                "Control the descent for better muscle engagement."
            ],
            "BreastStroke": [
                "Glide longer between strokes for efficiency.",
                "Keep your kick narrow and powerful.",
                "Time your arm pull with your breath.",
                "Keep your head in line with your spine."
            ],
            "PullUps": [
                "Pull your chest to the bar, not just your chin.",
                "Lower with control for full muscle engagement.",
                "Keep your core engaged throughout the movement.",
                "Maintain a steady rhythm for each repetition."
            ],
            "Unknown": [
                "Focus on controlled, smooth movements.",
                "Maintain proper form throughout the exercise.",
                "Keep a steady pace for better results.",
                "Engage the correct muscles for this movement."
            ]
        }
        
        # Get feedback options for the current exercise or use generic ones
        options = feedback_options.get(action, feedback_options["Unknown"])
        
        # Select a random feedback option
        import random
        feedback = random.choice(options)
        
        # Add to feedback history
        self.feedback_history.append(feedback)
        
        return feedback

def draw_coach_feedback(app, feedback, landmarks=None, sensor_data=None):
    """Draw AI coaching feedback on the right side of the 3D interface"""
    try:
        # Skip if feedback display is disabled
        if not hasattr(app, 'feedback_visible') or not app.feedback_visible:
            return
            
        # Get the 3D canvas
        canvas = app.colored_frame
        
        # Define position for the feedback box - right side of the interface
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        overlay_width = 250
        overlay_height = 300
        overlay_x = canvas_width - overlay_width - 10  # Right side
        overlay_y = 10  # Top position
        
        # Clear previous feedback
        canvas.delete("coach_feedback")
        
        # Draw background
        bg_color = "black"
        canvas.create_rectangle(
            overlay_x, overlay_y,
            overlay_x + overlay_width, overlay_y + overlay_height,
            fill=bg_color, stipple='gray50', outline='white',
            tags="coach_feedback"
        )
        
        # Draw title
        canvas.create_text(
            overlay_x + overlay_width//2, overlay_y + 15,
            text="AI Coach Feedback",
            fill='white', font=('Arial', 12, 'bold'),
            tags="coach_feedback"
        )
        
        # Draw exercise info if active
        if hasattr(app, 'exercise_active') and app.exercise_active and hasattr(app, 'ai_coach') and app.ai_coach:
            exercise = app.ai_coach.current_exercise
            canvas.create_text(
                overlay_x + overlay_width//2, overlay_y + 40,
                text=f"Exercise: {exercise}",
                fill='#00FFFF', font=('Arial', 10, 'bold'),
                tags="coach_feedback"
            )
            
            # Calculate session duration
            if hasattr(app, 'timer_start') and app.timer_start:
                duration = time.time() - app.timer_start
                mins = int(duration // 60)
                secs = int(duration % 60)
                
                canvas.create_text(
                    overlay_x + overlay_width//2, overlay_y + 60,
                    text=f"Duration: {mins:02d}:{secs:02d}",
                    fill='white', font=('Arial', 9),
                    tags="coach_feedback"
                )
            
            # Draw metrics
            if hasattr(app, 'exercise_metrics'):
                metrics_y = overlay_y + 85
                
                # Rep count - update from AI coach
                rep_count = app.ai_coach.rep_count.get(exercise, 0)
                app.exercise_metrics['rep_count'] = rep_count
                
                # Form score - update from AI coach if available
                if landmarks and app.ai_coach:
                    # Get action from action classifier
                    action = app.last_action if hasattr(app, 'last_action') else None
                    # Calculate form score
                    form_score = app.ai_coach.calculate_form_score(landmarks, action)
                    app.exercise_metrics['form_score'] = form_score
                
                # Draw metrics
                metrics = [
                    f"Reps: {app.exercise_metrics['rep_count']}",
                    f"Form Score: {app.exercise_metrics['form_score']:.0f}%",
                ]
                
                for metric in metrics:
                    canvas.create_text(
                        overlay_x + 20, metrics_y,
                        text=metric,
                        fill='white', font=('Arial', 9),
                        anchor=tk.W,
                        tags="coach_feedback"
                    )
                    metrics_y += 20
                
                # Form score visual indicator
                score = app.exercise_metrics['form_score']
                meter_width = 180
                meter_height = 6
                meter_x = overlay_x + (overlay_width - meter_width) // 2
                meter_y = metrics_y + 5
                
                # Determine color based on score
                if score > 80:
                    score_color = "#00FF00"  # Green for good form
                elif score > 50:
                    score_color = "#FFFF00"  # Yellow for okay form
                else:
                    score_color = "#FF0000"  # Red for poor form
                
                # Draw meter background
                canvas.create_rectangle(
                    meter_x, meter_y,
                    meter_x + meter_width, meter_y + meter_height,
                    fill='gray30', outline='gray50',
                    tags="coach_feedback"
                )
                
                # Draw filled form score meter
                fill_width = int(meter_width * (score / 100))
                canvas.create_rectangle(
                    meter_x, meter_y,
                    meter_x + fill_width, meter_y + meter_height,
                    fill=score_color, outline='',
                    tags="coach_feedback"
                )
                
                metrics_y += 20
        
        # Draw the actual feedback
        if feedback:
            # Create text with word wrapping
            text_y = overlay_y + (170 if hasattr(app, 'exercise_active') and app.exercise_active else 60)
            
            # Split feedback into lines for word wrapping
            words = feedback.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) <= 38:  # Character limit per line
                    current_line += (" " + word if current_line else word)
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw each line
            for line in lines:
                canvas.create_text(
                    overlay_x + overlay_width//2, text_y,
                    text=line,
                    fill='yellow', font=('Arial', 9),
                    width=overlay_width - 20,  # Text wrapping width
                    tags="coach_feedback"
                )
                text_y += 20
        
        # Add Analysis Data button below the coach feedback
        # Clear previous button
        canvas.delete("analysis_button")
        
        # Create button dimensions and position
        btn_width = 200
        btn_height = 30
        btn_x = overlay_x + (overlay_width - btn_width) // 2
        btn_y = overlay_y + overlay_height + 10
        
        # Draw button background with gradient effect
        canvas.create_rectangle(
            btn_x, btn_y,
            btn_x + btn_width, btn_y + btn_height,
            fill="#333333", outline='#666666',
            tags="analysis_button"
        )
        
        # Add highlight effect to top of button (for 3D look)
        canvas.create_line(
            btn_x + 1, btn_y + 1,
            btn_x + btn_width - 1, btn_y + 1,
            fill="#777777",
            tags="analysis_button"
        )
        
        # Draw button text
        canvas.create_text(
            btn_x + btn_width//2, btn_y + btn_height//2,
            text="Show Analysis Data",
            fill='white', font=('Arial', 10, 'bold'),
            tags="analysis_button"
        )
        
        # Add click binding for the button
        if hasattr(app, 'analysis_button_binding'):
            canvas.tag_unbind("analysis_button", "<Button-1>", app.analysis_button_binding)
        
        app.analysis_button_binding = canvas.tag_bind(
            "analysis_button", "<Button-1>", 
            lambda event: app.show_analysis_options() if hasattr(app, 'show_analysis_options') else None
        )
        
    except Exception as e:
        print(f"Error drawing coach feedback: {e}") 