"""
Gemini AI Coach for Exercise Feedback
Provides personalized AI coaching for exercises by analyzing pose data
and offering feedback to improve form and performance.
"""

import requests
import json
import numpy as np
import time
import os
import math
from collections import deque

# API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-exp-03-25:generateContent"
# Load API key from environment variable or config file
API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Replace with your API key

# Supported exercises and their corresponding metrics
SUPPORTED_EXERCISES = [
    "Squat", 
    "Push-up", 
    "Plank", 
    "Lunge", 
    "Bicep Curl",
    "Shoulder Press",
    "Jumping Jack"
]

# Exercise-specific metrics and ideal values
EXERCISE_METRICS = {
    "Squat": {
        "knee_angle": (90, 110),  # Ideal range in degrees
        "back_angle": (60, 90),   # Ideal range in degrees
        "symmetry": (0.9, 1.0),   # Similarity between left and right sides
        "depth": (0.4, 0.6)       # Relative to height
    },
    "Push-up": {
        "elbow_angle": (80, 100),
        "back_straightness": (0.85, 1.0),
        "head_position": (0.8, 1.0),
        "depth": (0.4, 0.6)
    },
    "Plank": {
        "back_straightness": (0.9, 1.0),
        "hip_position": (0.8, 1.0),
        "stability": (0.9, 1.0),
        "duration": (30, 120)  # Seconds
    },
    "Lunge": {
        "front_knee_angle": (85, 95),
        "back_knee_angle": (85, 95),
        "torso_upright": (0.85, 1.0),
        "balance": (0.85, 1.0)
    },
    "Bicep Curl": {
        "elbow_angle_range": (30, 160),
        "torso_stability": (0.9, 1.0),
        "wrist_position": (0.85, 1.0),
        "symmetry": (0.9, 1.0)
    },
    "Shoulder Press": {
        "elbow_angle_range": (50, 170),
        "shoulder_alignment": (0.85, 1.0),
        "torso_stability": (0.9, 1.0),
        "symmetry": (0.9, 1.0)
    },
    "Jumping Jack": {
        "arm_extension": (0.85, 1.0),
        "leg_spread": (0.8, 1.0),
        "synchronization": (0.9, 1.0),
        "rhythm": (0.85, 1.0)
    }
}

class GeminiCoach:
    """AI coach that analyzes pose data and provides exercise feedback using Gemini AI."""
    
    def __init__(self):
        """Initialize the AI coach."""
        self.exercise = None
        self.metrics = {}
        self.session_active = False
        self.session_start_time = None
        self.session_duration = 0
        self.rep_count = 0
        self.feedback_history = []
        self.landmark_history = deque(maxlen=30)  # Store last 30 frames of landmarks
        
        # For repetition detection
        self.last_rep_time = 0
        self.rep_positions = []  # Track positions during reps
        self.prev_phase = None
        
        # Metrics thresholds for feedback
        self.feedback_thresholds = {
            "poor": 0.4,
            "fair": 0.7,
            "good": 0.85,
            "excellent": 0.95
        }
        
        # Check if API key is available
        if not API_KEY:
            print("Warning: Gemini API key not found. Set GEMINI_API_KEY environment variable.")
    
    def set_exercise(self, exercise):
        """Set the current exercise for coaching."""
        if exercise in SUPPORTED_EXERCISES:
            self.exercise = exercise
            self.metrics = {metric: 0.0 for metric in EXERCISE_METRICS.get(exercise, {})}
            self.metrics["reps"] = 0
            self.rep_count = 0
            self.landmark_history.clear()
            self.feedback_history = []
            print(f"Exercise set to {exercise}")
            return True
        print(f"Unsupported exercise: {exercise}")
        return False
    
    def start_session(self):
        """Start a new exercise session."""
        if not self.exercise:
            print("No exercise selected")
            return False
        
        self.session_active = True
        self.session_start_time = time.time()
        self.last_rep_time = time.time()
        self.rep_count = 0
        self.feedback_history = []
        self.landmark_history.clear()
        self.metrics = {metric: 0.0 for metric in EXERCISE_METRICS.get(self.exercise, {})}
        self.metrics["reps"] = 0
        
        print(f"Started {self.exercise} session")
        return True
    
    def end_session(self):
        """End the current exercise session and provide final feedback."""
        if not self.session_active:
            return None
        
        self.session_active = False
        self.session_duration = time.time() - self.session_start_time
        
        # Generate final feedback using Gemini
        final_feedback = self._generate_final_feedback()
        
        print(f"Ended {self.exercise} session - {self.rep_count} reps completed")
        return final_feedback
    
    def update_metrics(self, landmarks):
        """Update exercise metrics based on pose landmarks."""
        if not self.session_active or not self.exercise:
            return None
        
        # Add landmarks to history
        self.landmark_history.append(landmarks)
        
        # Calculate exercise-specific metrics
        if self.exercise == "Squat":
            self._calculate_squat_metrics(landmarks)
        elif self.exercise == "Push-up":
            self._calculate_pushup_metrics(landmarks)
        elif self.exercise == "Plank":
            self._calculate_plank_metrics(landmarks)
        elif self.exercise == "Lunge":
            self._calculate_lunge_metrics(landmarks)
        elif self.exercise == "Bicep Curl":
            self._calculate_bicep_curl_metrics(landmarks)
        elif self.exercise == "Shoulder Press":
            self._calculate_shoulder_press_metrics(landmarks)
        elif self.exercise == "Jumping Jack":
            self._calculate_jumping_jack_metrics(landmarks)
        
        # Detect repetitions
        self._detect_repetition(landmarks)
        
        # Generate feedback if needed
        if self.rep_count % 3 == 0 and self.rep_count > 0:
            # Give feedback every 3 reps
            return self._generate_live_feedback()
        
        return None
    
    def get_metrics(self):
        """Get the current exercise metrics."""
        if not self.exercise:
            return {}
        return self.metrics
    
    def _calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = np.array([a['x'], a['y'], a['z']])
        b = np.array([b['x'], b['y'], b['z']])
        c = np.array([c['x'], c['y'], c['z']])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_distance(self, p1, p2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)
    
    def _normalize_metric(self, value, range_min, range_max):
        """Normalize a metric to a 0-1 scale based on ideal range."""
        if value < range_min:
            return max(0, 1 - (range_min - value) / range_min)
        elif value > range_max:
            return max(0, 1 - (value - range_max) / range_max)
        else:
            # Value is in ideal range
            return 1.0
    
    def _detect_repetition(self, landmarks):
        """Detect exercise repetitions based on landmark movements."""
        if not landmarks or len(landmarks) < 33:
            return
        
        # Different detection logic for each exercise type
        if self.exercise == "Squat":
            # Track hip position for squat
            hip_y = landmarks[23]['y']  # Left hip y-position
            
            # Determine phase (up or down)
            current_phase = "down" if hip_y > 0.6 else "up"
            
            # Detect rep when transitioning from down to up
            if self.prev_phase == "down" and current_phase == "up":
                # Ensure minimum time between reps to avoid double counting
                if time.time() - self.last_rep_time > 1.0:
                    self.rep_count += 1
                    self.metrics["reps"] = self.rep_count
                    self.last_rep_time = time.time()
            
            self.prev_phase = current_phase
            
        elif self.exercise == "Push-up":
            # Track shoulder position for push-up
            shoulder_y = (landmarks[11]['y'] + landmarks[12]['y']) / 2  # Average shoulder height
            
            # Determine phase (up or down)
            current_phase = "down" if shoulder_y > 0.7 else "up"
            
            # Detect rep when transitioning from down to up
            if self.prev_phase == "down" and current_phase == "up":
                if time.time() - self.last_rep_time > 1.0:
                    self.rep_count += 1
                    self.metrics["reps"] = self.rep_count
                    self.last_rep_time = time.time()
            
            self.prev_phase = current_phase
            
        elif self.exercise == "Bicep Curl":
            # Track wrist position relative to shoulder
            wrist_y = landmarks[15]['y']  # Right wrist
            shoulder_y = landmarks[11]['y']  # Right shoulder
            
            # Determine phase (up or down)
            current_phase = "up" if wrist_y < shoulder_y else "down"
            
            # Detect rep when transitioning from up to down
            if self.prev_phase == "up" and current_phase == "down":
                if time.time() - self.last_rep_time > 1.0:
                    self.rep_count += 1
                    self.metrics["reps"] = self.rep_count
                    self.last_rep_time = time.time()
            
            self.prev_phase = current_phase
        
        # Similar logic can be added for other exercises
    
    def _calculate_squat_metrics(self, landmarks):
        """Calculate metrics for squat exercise."""
        if len(landmarks) < 33:
            return
        
        # Calculate knee angle (angle between hip, knee, and ankle)
        right_knee_angle = self._calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        left_knee_angle = self._calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        
        # Calculate back angle (angle between shoulder, hip, and knee)
        back_angle = self._calculate_angle(landmarks[11], landmarks[24], landmarks[26])
        
        # Calculate symmetry (similarity between left and right knee angles)
        knee_symmetry = min(right_knee_angle, left_knee_angle) / max(right_knee_angle, left_knee_angle)
        
        # Calculate squat depth (hip height relative to knee height)
        hip_height = landmarks[24]['y']
        knee_height = landmarks[26]['y']
        relative_depth = 1 - (hip_height / knee_height)
        
        # Normalize metrics to 0-1 scale
        knee_range = EXERCISE_METRICS["Squat"]["knee_angle"]
        back_range = EXERCISE_METRICS["Squat"]["back_angle"]
        symmetry_range = EXERCISE_METRICS["Squat"]["symmetry"]
        depth_range = EXERCISE_METRICS["Squat"]["depth"]
        
        self.metrics["knee_angle"] = self._normalize_metric(avg_knee_angle, knee_range[0], knee_range[1])
        self.metrics["back_angle"] = self._normalize_metric(back_angle, back_range[0], back_range[1])
        self.metrics["symmetry"] = self._normalize_metric(knee_symmetry, symmetry_range[0], symmetry_range[1])
        self.metrics["depth"] = self._normalize_metric(relative_depth, depth_range[0], depth_range[1])
    
    def _calculate_pushup_metrics(self, landmarks):
        """Calculate metrics for push-up exercise."""
        if len(landmarks) < 33:
            return
        
        # Calculate elbow angle
        right_elbow_angle = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        left_elbow_angle = self._calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        avg_elbow_angle = (right_elbow_angle + left_elbow_angle) / 2
        
        # Calculate back straightness (alignment of shoulders, hips, and ankles)
        shoulder_mid = [(landmarks[11]['x'] + landmarks[12]['x'])/2, 
                         (landmarks[11]['y'] + landmarks[12]['y'])/2, 
                         (landmarks[11]['z'] + landmarks[12]['z'])/2]
        hip_mid = [(landmarks[23]['x'] + landmarks[24]['x'])/2, 
                    (landmarks[23]['y'] + landmarks[24]['y'])/2, 
                    (landmarks[23]['z'] + landmarks[24]['z'])/2]
        ankle_mid = [(landmarks[27]['x'] + landmarks[28]['x'])/2, 
                      (landmarks[27]['y'] + landmarks[28]['y'])/2, 
                      (landmarks[27]['z'] + landmarks[28]['z'])/2]
        
        # Create a line from shoulders to ankles and measure deviation of hips
        direction = [ankle_mid[0] - shoulder_mid[0], 
                     ankle_mid[1] - shoulder_mid[1], 
                     ankle_mid[2] - shoulder_mid[2]]
        length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        if length > 0:
            direction = [d/length for d in direction]
        
        # Calculate deviation of hips from the line
        t = ((hip_mid[0] - shoulder_mid[0]) * direction[0] + 
             (hip_mid[1] - shoulder_mid[1]) * direction[1] + 
             (hip_mid[2] - shoulder_mid[2]) * direction[2])
        
        closest_point = [shoulder_mid[0] + t * direction[0],
                         shoulder_mid[1] + t * direction[1],
                         shoulder_mid[2] + t * direction[2]]
        
        hip_deviation = math.sqrt((hip_mid[0] - closest_point[0])**2 + 
                                  (hip_mid[1] - closest_point[1])**2 + 
                                  (hip_mid[2] - closest_point[2])**2)
        
        back_straightness = 1 - min(1, hip_deviation * 10)  # Scale deviation
        
        # Calculate head position (alignment with back)
        nose = [landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']]
        
        head_direction = [nose[0] - shoulder_mid[0],
                         nose[1] - shoulder_mid[1],
                         nose[2] - shoulder_mid[2]]
        
        head_length = math.sqrt(head_direction[0]**2 + head_direction[1]**2 + head_direction[2]**2)
        if head_length > 0:
            head_direction = [d/head_length for d in head_direction]
        
        # Dot product with 'up' direction to measure head alignment
        up_direction = [0, -1, 0]  # Negative Y is 'up' in screen coordinates
        head_alignment = abs(head_direction[0] * up_direction[0] + 
                            head_direction[1] * up_direction[1] + 
                            head_direction[2] * up_direction[2])
        
        head_position = head_alignment
        
        # Calculate depth (how low the body goes)
        shoulder_height = (landmarks[11]['y'] + landmarks[12]['y']) / 2
        hip_height = (landmarks[23]['y'] + landmarks[24]['y']) / 2
        relative_height_diff = abs(shoulder_height - hip_height)
        
        # Normalize metrics
        elbow_range = EXERCISE_METRICS["Push-up"]["elbow_angle"]
        back_range = EXERCISE_METRICS["Push-up"]["back_straightness"]
        head_range = EXERCISE_METRICS["Push-up"]["head_position"]
        depth_range = EXERCISE_METRICS["Push-up"]["depth"]
        
        self.metrics["elbow_angle"] = self._normalize_metric(avg_elbow_angle, elbow_range[0], elbow_range[1])
        self.metrics["back_straightness"] = self._normalize_metric(back_straightness, back_range[0], back_range[1])
        self.metrics["head_position"] = self._normalize_metric(head_position, head_range[0], head_range[1])
        self.metrics["depth"] = self._normalize_metric(relative_height_diff, depth_range[0], depth_range[1])
    
    def _calculate_plank_metrics(self, landmarks):
        """Calculate metrics for plank exercise."""
        # Implementation would follow similar pattern as squat and push-up metrics
        if len(landmarks) < 33:
            return
            
        # Basic implementation - actual metrics would be more sophisticated
        self.metrics["back_straightness"] = 0.85
        self.metrics["hip_position"] = 0.9
        self.metrics["stability"] = 0.95
        
        # Update duration
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            self.metrics["duration"] = min(1.0, duration / 120.0)  # Normalize to max 120 seconds
    
    def _calculate_lunge_metrics(self, landmarks):
        """Calculate metrics for lunge exercise."""
        # Simplified implementation
        if len(landmarks) < 33:
            return
            
        # Basic placeholder implementation
        self.metrics["front_knee_angle"] = 0.85
        self.metrics["back_knee_angle"] = 0.8
        self.metrics["torso_upright"] = 0.9
        self.metrics["balance"] = 0.85
    
    def _calculate_bicep_curl_metrics(self, landmarks):
        """Calculate metrics for bicep curl exercise."""
        # Simplified implementation
        if len(landmarks) < 33:
            return
            
        # Basic placeholder implementation
        self.metrics["elbow_angle_range"] = 0.9
        self.metrics["torso_stability"] = 0.85
        self.metrics["wrist_position"] = 0.9
        self.metrics["symmetry"] = 0.8
    
    def _calculate_shoulder_press_metrics(self, landmarks):
        """Calculate metrics for shoulder press exercise."""
        # Simplified implementation
        if len(landmarks) < 33:
            return
            
        # Basic placeholder implementation
        self.metrics["elbow_angle_range"] = 0.85
        self.metrics["shoulder_alignment"] = 0.9
        self.metrics["torso_stability"] = 0.95
        self.metrics["symmetry"] = 0.9
    
    def _calculate_jumping_jack_metrics(self, landmarks):
        """Calculate metrics for jumping jack exercise."""
        # Simplified implementation
        if len(landmarks) < 33:
            return
            
        # Basic placeholder implementation
        self.metrics["arm_extension"] = 0.9
        self.metrics["leg_spread"] = 0.85
        self.metrics["synchronization"] = 0.9
        self.metrics["rhythm"] = 0.85
    
    def _generate_live_feedback(self):
        """Generate real-time feedback based on current metrics."""
        if not self.exercise or not self.metrics or not API_KEY:
            return "Exercise in progress. Continue performing the exercise."
        
        # Find the lowest scoring metrics to focus feedback on
        sorted_metrics = sorted([(k, v) for k, v in self.metrics.items() if k != "reps"], 
                               key=lambda x: x[1])
        
        # Take the two lowest scoring metrics
        focus_metrics = sorted_metrics[:2] if len(sorted_metrics) >= 2 else sorted_metrics
        
        # Prepare context for Gemini API
        metrics_text = ", ".join([f"{k}: {v:.2f}" for k, v in self.metrics.items() 
                                if k != "reps"])
        
        prompt = f"""
        You are a professional exercise coach giving feedback on a {self.exercise} exercise.
        
        Current metrics (0-1 scale where 1 is perfect):
        {metrics_text}
        
        Current repetition count: {self.rep_count}
        
        Please provide brief, specific feedback focusing on improving these areas: 
        {', '.join([f"{m[0]} ({m[1]:.2f})" for m in focus_metrics])}
        
        Keep feedback positive, encouraging, and concise (2-3 sentences maximum).
        """
        
        try:
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            if response:
                # Store this feedback
                self.feedback_history.append(response)
                return response
            else:
                # Fallback feedback if API fails
                return self._generate_fallback_feedback(focus_metrics)
                
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return self._generate_fallback_feedback(focus_metrics)
    
    def _generate_final_feedback(self):
        """Generate comprehensive feedback at the end of a session."""
        if not self.exercise or not self.metrics or not API_KEY:
            return f"Completed {self.exercise} session with {self.rep_count} repetitions."
        
        # Calculate average metrics across the session
        avg_metrics = {k: v for k, v in self.metrics.items() if k != "reps"}
        
        # Prepare context for Gemini API
        metrics_text = ", ".join([f"{k}: {v:.2f}" for k, v in avg_metrics.items()])
        
        duration_minutes = self.session_duration / 60
        
        prompt = f"""
        You are a professional exercise coach providing final feedback on a {self.exercise} session.
        
        Session summary:
        - Exercise: {self.exercise}
        - Duration: {duration_minutes:.1f} minutes
        - Repetitions completed: {self.rep_count}
        
        Average metrics (0-1 scale where 1 is perfect):
        {metrics_text}
        
        Please provide comprehensive feedback on this session:
        1. Highlight what was done well
        2. Identify 1-2 specific areas for improvement
        3. Include a brief tip for next time
        
        Keep feedback encouraging, specific, and actionable. Maximum 3-4 sentences.
        """
        
        try:
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            if response:
                return response
            else:
                # Fallback feedback if API fails
                return f"Completed {self.exercise} session with {self.rep_count} repetitions. Focus on maintaining proper form and gradually increasing intensity in future sessions."
                
        except Exception as e:
            print(f"Error generating final feedback: {e}")
            return f"Great job completing your {self.exercise} session with {self.rep_count} repetitions!"
    
    def _generate_fallback_feedback(self, focus_metrics):
        """Generate simple feedback without API when needed."""
        if not focus_metrics:
            return f"Continue your {self.exercise}. Completed {self.rep_count} repetitions so far."
        
        # Generic feedback messages based on metric names
        feedback_templates = {
            "knee_angle": "Focus on bending your knees to the proper depth.",
            "back_angle": "Try to maintain a straighter back position.",
            "symmetry": "Work on keeping your movement balanced on both sides.",
            "depth": "Pay attention to your range of motion.",
            "elbow_angle": "Watch your elbow positioning during the movement.",
            "back_straightness": "Keep your back straight throughout the exercise.",
            "head_position": "Remember to maintain proper head position.",
            "torso_upright": "Try to keep your torso more upright.",
            "balance": "Focus on your balance during the movement.",
            "torso_stability": "Minimize torso movement during the exercise.",
            "wrist_position": "Check your wrist alignment for better form.",
            "shoulder_alignment": "Keep your shoulders properly aligned.",
            "arm_extension": "Extend your arms fully during the movement.",
            "leg_spread": "Focus on proper leg positioning.",
            "synchronization": "Work on coordinating your movements.",
            "rhythm": "Try to maintain a consistent pace."
        }
        
        # Build feedback based on the lowest metrics
        feedback = f"You've completed {self.rep_count} repetitions. "
        
        for metric, value in focus_metrics:
            if metric in feedback_templates:
                feedback += feedback_templates[metric] + " "
                
        return feedback.strip()
    
    def _call_gemini_api(self, prompt):
        """Call the Gemini API with the provided prompt."""
        if not API_KEY:
            return None
            
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 200
            }
        }
        
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                data=json.dumps(data),
                timeout=5  # Increased timeout slightly
            )
            
            # Debug: Print complete response
            print(f"Gemini API Status Code: {response.status_code}")
            
            if response.status_code == 200:
                # Parse the response
                result = response.json()
                
                # Debug: Print full API response structure
                print(f"API Response Structure: {json.dumps(result, indent=2)}")
                
                # Try different response formats
                # Format 1: Standard Gemini format with candidates/parts
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        return content["parts"][0]["text"].strip()
                
                # Format 2: Alternative format with generations
                elif "generations" in result and len(result["generations"]) > 0:
                    if "text" in result["generations"][0]:
                        return result["generations"][0]["text"].strip()
                
                # Format 3: Direct content in response
                elif "content" in result:
                    if isinstance(result["content"], str):
                        return result["content"].strip()
                    elif isinstance(result["content"], dict) and "text" in result["content"]:
                        return result["content"]["text"].strip()
                
                # If no known format matched, try a simple fallback approach
                # Walk through the response to find text content
                def find_text_content(obj):
                    if isinstance(obj, str):
                        return obj
                    elif isinstance(obj, dict):
                        # Try common keys first
                        for key in ['content', 'text', 'message']:
                            if key in obj:
                                result = find_text_content(obj[key])
                                if result:
                                    return result
                        # Then try all values
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
                
                text_content = find_text_content(result)
                if text_content:
                    return text_content.strip()
                
                print(f"Could not extract text from API response")
            else:
                print(f"API error: {response.status_code} - {response.text}")
            
            return None
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None 