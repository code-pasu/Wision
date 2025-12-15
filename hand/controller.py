"""
Main hand gesture controller.

This module provides the HandGestureController class which:
- Orchestrates the full pipeline: capture -> track -> recognize -> act -> display
- Maps gestures to actions based on the current control mode
- Renders a real-time UI overlay showing mode, gesture, and controls

Entry points:
- HandGestureController().run()  -- Start the controller
- main()                         -- Convenience function
"""

import cv2
import numpy as np
import time
from typing import Optional

from .tracker import HandTracker
from .recognizer import GestureRecognizer
from .actions import ActionController
from .gestures import Gesture, ControlMode


class HandGestureController:
    """Main controller combining tracking, recognition, and actions."""
    
    # Gesture to action mappings per mode
    CURSOR_ACTIONS = {
        Gesture.INDEX_UP: 'move_cursor',
        Gesture.GRAB: 'drag',
        Gesture.PINCH_MIDDLE: 'left_click',
        Gesture.ROCK_SIGN: 'right_click',
        Gesture.CALL_ME: 'double_click',
        Gesture.PEACE_SIGN: 'scroll',
        Gesture.RING_CURL: 'middle_click',
        Gesture.MIDDLE_CURL: 'scroll_click',
    }
    
    SCROLL_ACTIONS = {
        Gesture.PEACE_SIGN: 'scroll',
        Gesture.INDEX_UP: 'move_cursor',
    }
    
    WINDOW_ACTIONS = {
        Gesture.INDEX_UP: 'move_cursor',
        Gesture.OPEN_PALM: 'maximize',
        Gesture.GRAB: 'minimize',
        Gesture.ROCK_SIGN: 'switch_window',
        Gesture.PINCH_MIDDLE: 'show_desktop',
        Gesture.CALL_ME: 'close_window',
        Gesture.PEACE_SIGN: 'screenshot',
    }
    
    MEDIA_ACTIONS = {
        Gesture.OPEN_PALM: 'play_pause',
        Gesture.INDEX_UP: 'next_track',
        Gesture.PEACE_SIGN: 'prev_track',
        Gesture.PINCH_MIDDLE: 'volume_up',
        Gesture.ROCK_SIGN: 'volume_down',
        Gesture.GRAB: 'mute',
    }
    
    def __init__(self):
        self.tracker = HandTracker()
        self.recognizer = GestureRecognizer(self.tracker)
        self.actions = ActionController()
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        
    def start_camera(self, camera_id: int = 0) -> bool:
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    def process_frame(self) -> Optional[tuple]:
        """Process one frame and return (frame, gesture)."""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process with hand tracker
        hand_detected = self.tracker.process(frame)
        
        if hand_detected:
            # Recognize gesture
            gesture = self.recognizer.recognize()
            
            # Draw landmarks
            frame = self.tracker.draw_landmarks(frame)
            
            # Handle OK sign for mode switching (requires both frames AND time)
            if gesture == Gesture.OK_SIGN and self.recognizer.is_gesture_ready(5, 0.3):
                self.actions.switch_mode()
            elif gesture != Gesture.OK_SIGN and gesture != Gesture.NONE:
                # Execute gesture actions based on mode
                self._execute_gesture_action(gesture)
            
            return (frame, gesture)
        
        # Reset smoother when hand is lost
        self.actions.reset_smoother()
        return (frame, Gesture.NONE)
    
    def _handle_cursor_movement(self, gesture: Gesture):
        """Handle cursor movement for INDEX_UP and L_SIGN gestures."""
        if gesture == Gesture.INDEX_UP or gesture == Gesture.L_SIGN:
            pos = self.tracker.get_landmark(self.tracker.INDEX_TIP)
            if pos:
                self.actions.move_cursor_relative(pos[0], pos[1])
                
                # L_SIGN (thumb + index) triggers left click with time-based check
                if gesture == Gesture.L_SIGN:
                    if self.recognizer.is_gesture_ready(2, 0.8):
                        self.actions.left_click()
    
    def _execute_gesture_action(self, gesture: Gesture):
        """Execute appropriate action for gesture based on current mode."""
        mode = self.actions.current_mode
        
        if mode == ControlMode.CURSOR:
            self._handle_cursor_mode(gesture)
        elif mode == ControlMode.SCROLL:
            self._handle_scroll_mode(gesture)
        elif mode == ControlMode.WINDOW:
            self._handle_window_mode(gesture)
        elif mode == ControlMode.MEDIA:
            self._handle_media_mode(gesture)
    
    def _handle_cursor_mode(self, gesture: Gesture):
        """Handle cursor mode gestures."""
        # Handle cursor movement (INDEX_UP moves, L_SIGN moves + clicks)
        self._handle_cursor_movement(gesture)
                
        if gesture == Gesture.PINCH_MIDDLE:
            if self.recognizer.is_gesture_ready(3, 0.35):
                self.actions.left_click()
                
        elif gesture == Gesture.ROCK_SIGN:
            if self.recognizer.is_gesture_ready(3, 0.82):
                self.actions.right_click()
                
        elif gesture == Gesture.CALL_ME:
            if self.recognizer.is_gesture_ready(5, 0.25):
                self.actions.double_click()
                
        elif gesture == Gesture.PEACE_SIGN:
            # Scroll requires time-based check to avoid accidental scrolls
            if self.recognizer.is_gesture_held_for(0.1):
                angle = self.tracker.get_peace_sign_angle()
                if angle is not None:
                    self.actions.scroll_by_angle(angle)
        
        elif gesture == Gesture.RING_CURL:
            if self.recognizer.is_gesture_ready(3, 0.2):
                self.actions.middle_click()
    
    def _handle_scroll_mode(self, gesture: Gesture):
        """Handle scroll mode gestures."""
        # Handle cursor movement (INDEX_UP moves, L_SIGN moves + clicks)
        self._handle_cursor_movement(gesture)
                
        if gesture == Gesture.PEACE_SIGN:
            # Scroll requires time-based check to avoid accidental scrolls
            if self.recognizer.is_gesture_held_for(0.1):
                angle = self.tracker.get_peace_sign_angle()
                if angle is not None:
                    self.actions.scroll_by_angle(angle)
    
    def _handle_window_mode(self, gesture: Gesture):
        """Handle window mode gestures."""
        # Handle cursor movement (INDEX_UP moves, L_SIGN moves + clicks)
        self._handle_cursor_movement(gesture)
                
        if gesture == Gesture.OPEN_PALM:
            if self.recognizer.is_gesture_ready(5, 1.5):
                self.actions.maximize_window()
                
        elif gesture == Gesture.GRAB:
            if self.recognizer.is_gesture_ready(5, 1.5):
                self.actions.minimize_window()
                
        elif gesture == Gesture.ROCK_SIGN:
            if self.recognizer.is_gesture_ready(3, 0.62):
                self.actions.switch_window()
                
        elif gesture == Gesture.PINCH_MIDDLE:
            if self.recognizer.is_gesture_ready(3, 0.52):
                self.actions.show_desktop()
                
        elif gesture == Gesture.CALL_ME:
            if self.recognizer.is_gesture_ready(8, 0.5):
                self.actions.close_window()
                
        elif gesture == Gesture.PEACE_SIGN:
            if self.recognizer.is_gesture_ready(5, 0.83):
                self.actions.take_screenshot()
    
    def _handle_media_mode(self, gesture: Gesture):
        """Handle media mode gestures."""
        # Handle cursor movement (INDEX_UP moves, L_SIGN moves + clicks)
        self._handle_cursor_movement(gesture)
        
        if gesture == Gesture.OPEN_PALM:
            if self.recognizer.is_gesture_ready(3, 0.2):
                self.actions.play_pause()
                
        elif gesture == Gesture.PEACE_SIGN:
            if self.recognizer.is_gesture_ready(3, 0.2):
                self.actions.prev_track()
                
        elif gesture == Gesture.PINCH_MIDDLE:
            # Volume control with minimal delay for responsiveness
            if self.recognizer.is_gesture_held_for(0.1):
                self.actions.volume_up()
            
        elif gesture == Gesture.ROCK_SIGN:
            # Volume control with minimal delay for responsiveness
            if self.recognizer.is_gesture_held_for(0.1):
                self.actions.volume_down()
            
        elif gesture == Gesture.GRAB:
            if self.recognizer.is_gesture_ready(5, 0.3):
                self.actions.mute()
    
    def draw_ui(self, frame, gesture: Gesture):
        """Draw UI overlay on frame with gesture mappings."""
        h, w = frame.shape[:2]
        
        # Create a side panel for text (black background)
        panel_width = 250
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        
        # Draw mode indicator on panel
        mode = self.actions.current_mode
        mode_colors = {
            ControlMode.CURSOR: (255, 100, 100),   # Blue
            ControlMode.SCROLL: (100, 255, 100),   # Green
            ControlMode.WINDOW: (100, 100, 255),   # Red
            ControlMode.MEDIA: (255, 255, 100),    # Cyan
        }
        color = mode_colors.get(mode, (255, 255, 255))
        
        # Mode header
        cv2.rectangle(panel, (5, 5), (panel_width - 5, 50), color, -1)
        cv2.putText(panel, f"Mode: {mode.name}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Current gesture
        cv2.putText(panel, f"Gesture: {gesture.name}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Gesture mappings for current mode
        y = 105
        cv2.putText(panel, "CONTROLS:", (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 25
        
        # Get mappings for current mode
        mappings = self._get_mode_mappings(mode)
        for gesture_name, action in mappings:
            # Highlight if this is current gesture
            if gesture.name == gesture_name:
                cv2.rectangle(panel, (5, y - 12), (panel_width - 5, y + 5), (50, 100, 50), -1)
                text_color = (0, 255, 0)
            else:
                text_color = (200, 200, 200)
            
            cv2.putText(panel, f"{gesture_name}:", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
            cv2.putText(panel, action, (120, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
            y += 18
        
        # Mode switch instruction
        y += 10
        cv2.putText(panel, "OK_SIGN: Switch Mode", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
        
        # Drag indicator
        if self.actions.is_dragging:
            cv2.rectangle(panel, (5, h - 40), (panel_width - 5, h - 10), (0, 0, 200), -1)
            cv2.putText(panel, "DRAGGING", (60, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Quit instruction at bottom
        cv2.putText(panel, "Press Q to quit", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Combine frame and panel
        combined = np.hstack([frame, panel])
        
        return combined
    
    def _get_mode_mappings(self, mode: ControlMode):
        """Get gesture to action mappings for a mode."""
        # Common: INDEX_UP + Thumb = Left Click (all modes)
        common_note = "+Thumb: Click"
        
        if mode == ControlMode.CURSOR:
            return [
                ("INDEX_UP", f"Move ({common_note})"),
                ("PINCH_MIDDLE", "Left Click"),
                ("ROCK_SIGN", "Right Click"),
                ("CALL_ME", "Double Click"),
                ("PEACE_SIGN", "Scroll (angle)"),
                ("RING_CURL", "Middle Click"),
            ]
        elif mode == ControlMode.SCROLL:
            return [
                ("INDEX_UP", f"Move ({common_note})"),
                ("PEACE_SIGN", "Scroll (V=Up H=Down)"),
            ]
        elif mode == ControlMode.WINDOW:
            return [
                ("INDEX_UP", f"Move ({common_note})"),
                ("OPEN_PALM", "Maximize"),
                ("GRAB", "Minimize"),
                ("ROCK_SIGN", "Switch Window"),
                ("PINCH_MIDDLE", "Show Desktop"),
                ("CALL_ME", "Close Window"),
                ("PEACE_SIGN", "Screenshot"),
            ]
        elif mode == ControlMode.MEDIA:
            return [
                ("INDEX_UP", f"Move ({common_note})"),
                ("OPEN_PALM", "Play/Pause"),
                ("PEACE_SIGN", "Prev Track"),
                ("PINCH_MIDDLE", "Volume Up"),
                ("ROCK_SIGN", "Volume Down"),
                ("GRAB", "Mute"),
            ]
        return []
        
        return frame
    
    def run(self):
        """Main loop."""
        if not self.start_camera():
            print("Failed to open camera!")
            return
        
        print("Hand Gesture Controller Started")
        print("OK Sign = Switch Mode | Q = Quit")
        print(f"Current Mode: {self.actions.current_mode.name}")
        
        self.running = True
        
        try:
            while self.running:
                result = self.process_frame()
                
                if result is None:
                    continue
                
                frame, gesture = result
                frame = self.draw_ui(frame, gesture)
                
                cv2.imshow('Hand Gesture Controller', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.tracker.release()
        cv2.destroyAllWindows()


def main():
    """Entry point."""
    controller = HandGestureController()
    controller.run()


if __name__ == '__main__':
    main()
