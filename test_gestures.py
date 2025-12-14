"""
Gesture Testing Utility

This file displays all detected gestures in real-time for testing
and calibration purposes. It shows:
- Current detected gesture
- Finger states (extended/curled)
- Joint angles
- Gesture stability
- Detection confidence
"""

import cv2
import numpy as np
import time
from hand.tracker import HandTracker
from hand.recognizer import GestureRecognizer
from hand.gestures import Gesture, GESTURE_DESCRIPTIONS


def draw_finger_status(panel, tracker, start_y=80):
    """Draw finger extension and curl status on panel."""
    fingers = tracker.get_finger_states()
    curled = tracker.get_finger_curl_states()
    distances = tracker.get_finger_distances()
    
    x = 10
    y = start_y
    
    cv2.putText(panel, "FINGER STATUS:", (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 22
    
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for name in finger_names:
        extended = fingers.get(name, False)
        is_curled = curled.get(name, False) if name != 'thumb' else False
        
        if extended:
            status = "EXTENDED"
            color = (0, 255, 0)
        elif is_curled:
            status = "CURLED"
            color = (0, 0, 255)
        else:
            status = "partial"
            color = (0, 165, 255)
        
        cv2.putText(panel, f"  {name.capitalize():8s} {status}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18
    
    # Draw distances
    y += 8
    cv2.putText(panel, "DISTANCES (thumb):", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 22
    
    for name, dist in distances.items():
        short_name = name.replace('thumb_', '')
        # Highlight if pinch detected
        color = (0, 255, 255) if dist < 0.06 else (200, 200, 200)
        cv2.putText(panel, f"  {short_name:8s} {dist:.3f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18
    
    return y


def draw_gesture_list(panel, current_gesture, recognizer, start_y):
    """Draw list of all gestures with current one highlighted."""
    x = 10
    y = start_y + 10
    
    cv2.putText(panel, "ALL GESTURES:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 22
    
    for gesture in Gesture:
        if gesture == Gesture.NONE:
            continue
            
        is_current = gesture == current_gesture
        
        # Background for current gesture
        if is_current:
            cv2.rectangle(panel, (x - 2, y - 12), (240, y + 4), (0, 100, 0), -1)
        
        color = (0, 255, 0) if is_current else (150, 150, 150)
        marker = ">>" if is_current else "  "
        
        text = f"{marker} {gesture.name}"
        cv2.putText(panel, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18
    
    return y


def draw_stability_bar(panel, recognizer, y):
    """Draw gesture stability progress bar."""
    x = 10
    y += 10
    
    cv2.putText(panel, "STABILITY:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    bar_x = x + 80
    bar_width = 120
    
    # Background
    cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_width, y + 2), (50, 50, 50), -1)
    
    # Fill based on stable frames
    if recognizer.current_state:
        stable_frames = min(recognizer.current_state.stable_frames, 10)
        fill_width = int((stable_frames / 10) * bar_width)
        
        # Color gradient from red to green
        r = int(255 * (1 - stable_frames / 10))
        g = int(255 * (stable_frames / 10))
        
        cv2.rectangle(panel, (bar_x, y - 10), (bar_x + fill_width, y + 2), (0, g, r), -1)
        
        # Frame count
        cv2.putText(panel, f"{stable_frames}", (bar_x + bar_width + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return y + 20


def draw_peace_angle(panel, tracker, y):
    """Draw peace sign angle indicator."""
    x = 10
    angle = tracker.get_peace_sign_angle()
    
    cv2.putText(panel, "PEACE ANGLE:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if angle is not None:
        # Draw angle value
        angle_text = f"{angle:.1f}"
        color = (0, 255, 0) if abs(angle) > 30 else (200, 200, 200)
        cv2.putText(panel, angle_text, (x + 100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw direction indicator
        if angle > 30:
            cv2.putText(panel, "UP", (x + 150, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        elif angle < -30:
            cv2.putText(panel, "DOWN", (x + 150, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        cv2.putText(panel, "N/A", (x + 100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    return y + 25


def draw_description(panel, gesture, y, panel_height):
    """Draw gesture description at bottom."""
    if gesture != Gesture.NONE:
        desc = GESTURE_DESCRIPTIONS.get(gesture, "")
        cv2.putText(panel, "Description:", (10, panel_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(panel, desc, (10, panel_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def main():
    """Main function for gesture testing."""
    print("=" * 50)
    print("GESTURE TESTING UTILITY")
    print("=" * 50)
    print("This utility shows all detected gestures in real-time")
    print("No actions are performed - just for testing detection")
    print("Press Q or ESC to quit")
    print("=" * 50)
    
    # Initialize tracker
    tracker = HandTracker(
        detection_confidence=0.7,
        tracking_confidence=0.7
    )
    recognizer = GestureRecognizer(tracker)
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Camera opened successfully!")
    print("Starting gesture detection...")
    
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Panel width for text
    panel_width = 250
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Create dark panel for text
            panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
            panel[:] = (30, 30, 30)  # Dark gray background
            
            # Process frame
            hand_detected = tracker.process(frame)
            gesture = Gesture.NONE
            
            if hand_detected:
                # Recognize gesture
                gesture = recognizer.recognize()
                
                # Draw landmarks on camera frame
                frame = tracker.draw_landmarks(frame)
            
            # Draw header on panel
            if gesture != Gesture.NONE:
                cv2.rectangle(panel, (5, 5), (panel_width - 5, 35), (0, 100, 0), -1)
                cv2.putText(panel, gesture.name, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.rectangle(panel, (5, 5), (panel_width - 5, 35), (100, 0, 0), -1)
                text = "NO HAND" if not hand_detected else "NONE"
                cv2.putText(panel, text, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
            
            # Draw FPS
            cv2.putText(panel, f"FPS: {fps}", (panel_width - 70, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw separator line
            cv2.line(panel, (5, 45), (panel_width - 5, 45), (100, 100, 100), 1)
            
            if hand_detected:
                # Draw finger status
                y = draw_finger_status(panel, tracker, start_y=60)
                
                # Draw gesture list
                y = draw_gesture_list(panel, gesture, recognizer, start_y=y)
                
                # Draw stability bar
                y = draw_stability_bar(panel, recognizer, y)
                
                # Draw peace sign angle
                y = draw_peace_angle(panel, tracker, y)
                
                # Draw description at bottom
                draw_description(panel, gesture, y, h)
            else:
                # Just show gesture list
                draw_gesture_list(panel, Gesture.NONE, recognizer, start_y=60)
            
            # Draw quit instruction at very bottom
            cv2.putText(panel, "Press Q to quit", (10, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Combine frame and panel side by side
            combined = np.hstack([frame, panel])
            
            # Show combined window
            cv2.imshow('Gesture Testing', combined)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
    finally:
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()
        print("Gesture testing utility closed.")
        cv2.destroyAllWindows()
        print("Gesture testing utility closed.")


if __name__ == '__main__':
    main()
