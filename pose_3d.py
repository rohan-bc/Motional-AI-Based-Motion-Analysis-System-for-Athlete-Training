import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math

class Pose3DVisualizer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            enable_segmentation=True,
            smooth_landmarks=True
        )
        
        # Initialize PyGame and OpenGL
        pygame.init()
        self.display = (800, 600)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption('3D Pose Estimation')
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 10, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up camera
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, -1.0, -5)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Rotation angles
        self.rotation_x = 0
        self.rotation_y = 0
        
        # Colors for different body parts
        self.colors = {
            'torso': (0.0, 0.8, 0.0),      # Green
            'left_arm': (0.8, 0.0, 0.0),    # Red
            'right_arm': (0.0, 0.0, 0.8),   # Blue
            'left_leg': (0.8, 0.8, 0.0),    # Yellow
            'right_leg': (0.0, 0.8, 0.8),   # Cyan
            'head': (0.8, 0.4, 0.0)         # Orange
        }
        
        # Body segments for SMPL-like model
        self.body_segments = {
            'torso': [(11, 12), (12, 24), (24, 23), (23, 11)],
            'left_arm': [(11, 13), (13, 15)],
            'right_arm': [(12, 14), (14, 16)],
            'left_leg': [(23, 25), (25, 27)],
            'right_leg': [(24, 26), (26, 28)],
            'head': [(0, 1), (1, 4)]
        }

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

    def draw_ground_grid(self):
        """Draw a reference grid on the ground"""
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i/2, -2, -10)
            glVertex3f(i/2, -2, 10)
            glVertex3f(-10, -2, i/2)
            glVertex3f(10, -2, i/2)
        glEnd()

    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Read camera frame
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Convert to RGB and process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            # Clear screen and setup view
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            glTranslatef(0.0, -1.0, -5)
            
            # Apply rotation
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 1, 0)
            self.rotation_y += 0.5  # Continuous rotation
            
            # Draw ground grid
            self.draw_ground_grid()
            
            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                
                # Draw SMPL-like model
                for segment, connections in self.body_segments.items():
                    glColor3f(*self.colors[segment])
                    
                    for start_idx, end_idx in connections:
                        if (landmarks[start_idx].visibility > 0.5 and 
                            landmarks[end_idx].visibility > 0.5):
                            
                            start_point = np.array([
                                landmarks[start_idx].x,
                                landmarks[start_idx].y,
                                landmarks[start_idx].z
                            ]) * 2
                            
                            end_point = np.array([
                                landmarks[end_idx].x,
                                landmarks[end_idx].y,
                                landmarks[end_idx].z
                            ]) * 2
                            
                            # Draw limb cylinder
                            self.draw_cylinder(start_point, end_point)
                            
                            # Draw joints
                            glColor3f(0.9, 0.9, 0.9)
                            self.draw_sphere(start_point)
                            self.draw_sphere(end_point)
            
            pygame.display.flip()
            clock.tick(60)
        
        self.cap.release()
        pygame.quit()

def main():
    visualizer = Pose3DVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main() 