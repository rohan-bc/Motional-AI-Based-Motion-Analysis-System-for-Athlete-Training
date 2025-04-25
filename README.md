# Motional: AI-Based Motion Analysis System for Athlete Training

## 1. Project Overview
![Screenshot 2025-04-06 003647](https://github.com/user-attachments/assets/5ac79a41-d203-4fe0-b94c-c86f03cda2f8)

Motional is an AI-powered motion analysis system designed to help athletes train effectively by analyzing their movements using RGB camera video and wearable sensor data. The system tracks key motion patterns, evaluates exercise accuracy, and provides real-time feedback to optimize performance.

The platform combines computer vision, machine learning, and sensor fusion technologies to deliver comprehensive motion analysis for various sports and exercises. By integrating pose estimation with IMU sensor data, Motional provides a complete picture of an athlete's motion dynamics.

### Key Features

- **Real-time Pose Estimation**: Tracks body positions and movements through RGB camera input
- **Sensor Data Analysis**: Integrates with Phyphox mobile sensors for accelerometer and gyroscope readings
- **3D Motion Visualization**: Renders body movements in a 3D environment for detailed analysis
- **Action Classification**: Identifies specific exercises and sports movements using ML models
- **AI Coaching**: Provides real-time feedback and technique correction using advanced AI models
- **Performance Metrics**: Tracks and displays movement data for qualitative assessment
- **User-Friendly Interface**: Interactive controls for seamless experience
- **Enhanced Motion Analysis**: Bar graphs showing rate of change for sensor data with real-time metrics

## 2. Workflow & Implementation

### Phase 1: Data Acquisition (Capturing Motion)
- **Initialize System**: Start motion tracking setup with camera and sensors
- **Capture Video (RGB Camera)**: Record the athlete's exercise movements through webcam
- **Collect Sensor Data**: Gather gyroscope and accelerometer data from connected mobile device
- **Store Data for Processing**: Save raw data for further analysis

### Phase 2: Pose Estimation (Detecting Movement)
- **MediaPipe Integration**: Utilize Google's MediaPipe framework for human pose detection
- **Landmark Detection**: Identify and track 33 body landmarks in real-time
- **Pose Calculations**: Calculate joint angles and body positions
- **3D Pose Rendering**: Convert 2D landmarks to 3D visualizations with OpenGL

### Phase 3: Feature Extraction & Motion Analysis (Understanding Movement)
- **Motion Pattern Recognition**: Identify movement patterns and exercise types
- **Track Key Joints**: Monitor motion of essential body parts over time
- **Sensor Data Fusion**: Combine camera-based pose data with sensor readings
- **Motion Analysis Algorithms**: Apply algorithms to quantify movement quality
- **Visual Data Representation**: Bar graphs showing motion intensity and rate of change

### Phase 4: Exercise Classification & Feedback (Evaluating Movement)
- **ML-Based Classification**: Classify exercises using ConvLSTM neural networks
- **Performance Evaluation**: Compare athlete's movement with reference patterns
- **Generate AI Feedback**: Use LLM-based coaching system to provide technique advice
- **Display Results**: Present comprehensive analysis through the UI
- **Adjustable Parameters**: Control detection rate and confidence thresholds

## 3. Technical Implementation

### Technologies Used

- **Python**: Core programming language
- **OpenCV**: Computer vision operations and video processing
- **MediaPipe**: Real-time pose estimation
- **TensorFlow**: ML models for action classification
- **PyGame/OpenGL**: 3D visualization interface
- **Tkinter**: GUI framework
- **AI Coaching**: Support for multiple AI providers (Gemini, DeepSeek) via OpenRouter
- **Phyphox Integration**: Mobile sensor connectivity

### System Architecture

- **Modular Design**: Separate components for video capture, pose estimation, sensor data, analysis, and rendering
- **Multi-threaded Processing**: Parallel processing for responsive UI and real-time analysis
- **Event-Driven Updates**: UI updates based on new data from sensors and camera
- **Dynamic 3D Rendering**: Real-time 3D visualization using OpenGL
- **Fallback Systems**: API rate limit detection and graceful degradation

### ML Models

- **Action Classification**: ConvLSTM-based model trained on sports action datasets
- **Exercise Recognition**: Custom-trained model for specific exercise patterns
- **AI Coaching**: Integration with multiple AI providers for personalized feedback

## 4. User Guide

### System Requirements

- Windows 10 or later / macOS / Linux
- Webcam or compatible camera
- Python 3.8+ with required packages
- Mobile device with Phyphox app (for sensor data)

### Setup Instructions

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Configure Sensor Connection**:
   - Install Phyphox app on your mobile device
   - Open gyroscope experiment in Phyphox
   - Enable remote access and note the IP address
   - Enter the IP and port in the Sensor Controls panel

3. **Launch Application**:
   ```
   python pose_estimation.py
   ```

### Using the Interface

- **Main Window**: Displays camera feed, 2D skeleton, and 3D visualization
- **Controls Panel**: Access to various settings and features
- **View Controls**: Adjust 3D view, toggle background, reset view
- **Sensor Controls**: Connect to Phyphox sensors, adjust settings
- **Action Classification**: Set detection rate and confidence threshold
- **AI Coaching**: Select exercises, start/end sessions, view feedback
- **Analysis Data**: View and analyze motion data with bar graphs

## 5. Feature Details

### 3D Visualization

The 3D visualization provides a comprehensive view of the user's movements with:
- Skeletal rendering with joint connections
- Color-coded joint visualization
- Interactive camera controls (pan, zoom, rotate)
- Reference grid and platform for spatial orientation
- 90-degree rotation option for different perspectives

### Sensor Data Visualization

Sensor data is visualized through multiple displays:
- 2D time-series plots of gyroscope and accelerometer data
- Bar graphs showing rate of change for each axis (x, y, z)
- Motion analysis metrics (max/avg rotation, acceleration)
- Real-time magnitude indicators
- Color-coded visualization for easy interpretation

### Action Classification

The action classification system can identify various sports movements:
- TennisSwing, BoxingPunchingBag, Diving, Archery, Basketball
- LongJump, JugglingBalls, PushUps, BreastStroke, PullUps
- Configurable detection rate and confidence threshold
- Visual confidence meter
- Adjustable detection sensitivity

### AI Coach

The AI coaching system provides:
- Exercise-specific form feedback
- Real-time performance assessment
- Technique correction suggestions
- Rep counting for exercises
- Progress tracking
- Support for multiple AI providers with fallback capability

## 6. Recent Updates

- **Optimized UI Layout**: Adjusted size and position of motion analysis overlay to prevent overlap with action classification
- **Enhanced Motion Visualization**: Added bar graphs showing rate of change for sensor data
- **Improved Bar Graph Visualization**: Color-coded bars for x, y, z axes with real-time values
- **Multi-Provider AI Support**: Added support for different AI providers with fallback capability
- **API Rate Limit Handling**: Added graceful handling of API rate limits
- **View Controls**: Added 90-degree rotation option for better perspective
- **Performance Optimizations**: Improved frame rate and responsiveness
- **Sensor Reconnection**: Enhanced sensor reconnection capabilities with progressive retry intervals

## 7. Future Enhancements

- **Enhanced Exercise Library**: Additional sport-specific movements and exercises
- **Improved ML Models**: More accurate and responsive classification
- **Mobile App Integration**: Dedicated mobile application
- **Cloud Analytics**: Online storage and analysis of training data
- **Multi-Person Tracking**: Support for analyzing multiple athletes simultaneously

## 8. Contributors

- ROHAN BC - 21BRS1016

## 9. License

[Specify License Information] 
