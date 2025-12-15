"""
Gesture recognition from hand tracking data.

This module provides the GestureRecognizer class which:
- Analyzes finger states and distances to classify gestures
- Uses priority-ordered detection to avoid misclassification
- Tracks gesture stability (consecutive frames) before triggering actions

Gestures are checked from most specific to most general.
"""

import time
from typing import Optional
from .tracker import HandTracker
from .gestures import Gesture, GestureState


class GestureRecognizer:
    """Recognizes gestures from hand tracking data."""
    
    # Thresholds
    OK_SIGN_THRESHOLD = 0.05  # Tighter threshold for OK sign
    PINCH_THRESHOLD = 0.06
    STABILITY_FRAMES = 3
    MIN_GESTURE_DURATION = 0.15  # Minimum seconds a gesture must be held (default 150ms)
    
    def __init__(self, tracker: HandTracker, min_gesture_duration: Optional[float] = None):
        self.tracker = tracker
        self.current_state: Optional[GestureState] = None
        self.frame_count = 0
        # Allow custom duration threshold
        if min_gesture_duration is not None:
            self.min_gesture_duration = min_gesture_duration
        else:
            self.min_gesture_duration = self.MIN_GESTURE_DURATION
        
    def recognize(self) -> Gesture:
        """Recognize current gesture with priority ordering."""
        if self.tracker.landmarks is None:
            return self._update_state(Gesture.NONE)
        
        # Get finger states
        fingers = self.tracker.get_finger_states()
        curled = self.tracker.get_finger_curl_states()
        distances = self.tracker.get_finger_distances()
        
        # Priority-ordered detection (most specific first)
        
        # 1. OK sign (thumb-index circle, others extended) - strict check
        if self._is_ok_sign(fingers, distances):
            return self._update_state(Gesture.OK_SIGN)
        
        # 2. Call me (thumb + pinky, others curled)
        if self._is_call_me(fingers, curled):
            return self._update_state(Gesture.CALL_ME)
        
        # 3. L sign (thumb + index extended, others curled) - check BEFORE rock sign
        if self._is_l_sign(fingers, curled):
            return self._update_state(Gesture.L_SIGN)
        
        # 4. Rock sign (index + pinky extended, middle+ring curled, NO thumb)
        if self._is_rock_sign(fingers, curled):
            return self._update_state(Gesture.ROCK_SIGN)
        
        # 5. Pinch middle (thumb-middle close)
        if self._is_pinch_middle(fingers, distances):
            return self._update_state(Gesture.PINCH_MIDDLE)
        
        # 6. Peace sign (index + middle extended)
        if self._is_peace_sign(fingers, curled):
            return self._update_state(Gesture.PEACE_SIGN)
        
        # 7. Single curl gestures (specific finger curled, others extended)
        if self._is_ring_curl(fingers, curled):
            return self._update_state(Gesture.RING_CURL)
        
        if self._is_middle_curl(fingers, curled):
            return self._update_state(Gesture.MIDDLE_CURL)
        
        if self._is_pinky_curl(fingers, curled):
            return self._update_state(Gesture.PINKY_CURL)
        
        # 8. Open palm (all extended)
        if self._is_open_palm(fingers):
            return self._update_state(Gesture.OPEN_PALM)
        
        # 9. Index up (index extended only, no thumb)
        if self._is_index_up(fingers):
            return self._update_state(Gesture.INDEX_UP)
        
        # 10. Grab (closed/semi-closed hand - merged fist and grab)
        if self._is_grab(fingers, curled):
            return self._update_state(Gesture.GRAB)
        
        return self._update_state(Gesture.NONE)
    
    def _is_ok_sign(self, fingers: dict, distances: dict) -> bool:
        """Thumb and index forming circle, other fingers extended."""
        if 'thumb_index' not in distances:
            return False
        
        # Strict: thumb and index must be very close, and all other fingers extended
        return (
            distances['thumb_index'] < self.OK_SIGN_THRESHOLD and
            fingers['middle'] and
            fingers['ring'] and
            fingers['pinky'] and
            # Additional check: thumb_middle should be further than thumb_index
            distances.get('thumb_middle', 1.0) > distances['thumb_index'] * 1.5
        )
    
    def _is_l_sign(self, fingers: dict, curled: dict) -> bool:
        """Thumb and index extended (L shape), others curled.
        
        Stricter detection requiring fully extended thumb:
        - Thumb must be clearly extended away from palm
        - Index must be extended
        - Other fingers must be curled
        """
        if self.tracker.landmarks is None:
            return False
        
        # Basic finger state checks
        basic_check = (
            fingers['thumb'] and
            fingers['index'] and
            curled['middle'] and
            curled['ring'] and
            curled['pinky']
        )
        
        if not basic_check:
            return False
        
        # Additional strict check: thumb must be FULLY extended
        thumb_tip = self.tracker.get_landmark(self.tracker.THUMB_TIP)
        thumb_ip = self.tracker.get_landmark(self.tracker.THUMB_IP)
        thumb_mcp = self.tracker.get_landmark(self.tracker.THUMB_MCP)
        index_mcp = self.tracker.get_landmark(self.tracker.INDEX_MCP)
        wrist = self.tracker.get_landmark(self.tracker.WRIST)
        
        if None in (thumb_tip, thumb_ip, thumb_mcp, index_mcp, wrist):
            return basic_check
        
        # Calculate thumb angle at IP joint - must be very straight (> 155 degrees)
        thumb_angle = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        if thumb_angle < 155:
            return False
        
        # Thumb tip must be far from index MCP (thumb extended outward)
        thumb_to_index_mcp = (
            (thumb_tip[0] - index_mcp[0])**2 +  # type: ignore
            (thumb_tip[1] - index_mcp[1])**2  # type: ignore
        )**0.5
        
        # Require minimum distance (thumb clearly extended, not just slightly out)
        if thumb_to_index_mcp < 0.12:
            return False
        
        # Thumb tip should also be away from wrist (not curled back)
        thumb_to_wrist = (
            (thumb_tip[0] - wrist[0])**2 +  # type: ignore
            (thumb_tip[1] - wrist[1])**2  # type: ignore
        )**0.5
        
        # Thumb should be extended away from wrist
        if thumb_to_wrist < 0.15:
            return False
        
        return True
    
    def _calculate_angle(self, p1, p2, p3) -> float:
        """Calculate angle at p2 between p1-p2-p3."""
        import numpy as np
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 180.0
            
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
    
    def _is_rock_sign(self, fingers: dict, curled: dict) -> bool:
        """Index and pinky extended, middle and ring curled, NO thumb.
        
        Improved detection with additional checks:
        - Thumb must be curled/tucked (not extended)
        - Middle and ring must be clearly curled
        - Index and pinky must be clearly extended
        """
        if self.tracker.landmarks is None:
            return False
        
        # Basic finger state checks
        basic_check = (
            not fingers['thumb'] and  # Thumb must NOT be extended
            fingers['index'] and 
            fingers['pinky'] and
            curled['middle'] and
            curled['ring']
        )
        
        if not basic_check:
            return False
        
        # Additional check: verify thumb is actually tucked/curled near palm
        thumb_tip = self.tracker.get_landmark(self.tracker.THUMB_TIP)
        index_mcp = self.tracker.get_landmark(self.tracker.INDEX_MCP)
        pinky_mcp = self.tracker.get_landmark(self.tracker.PINKY_MCP)
        
        if thumb_tip is None or index_mcp is None or pinky_mcp is None:
            return basic_check
        
        # Thumb tip should be relatively close to palm (near index MCP)
        thumb_to_index_mcp = ((thumb_tip[0] - index_mcp[0])**2 + (thumb_tip[1] - index_mcp[1])**2)**0.5
        
        # Thumb should be tucked (distance < 0.12 normalized)
        thumb_tucked = thumb_to_index_mcp < 0.12
        
        # Additional: Check that index and pinky tips are spread apart
        index_tip = self.tracker.get_landmark(self.tracker.INDEX_TIP)
        pinky_tip = self.tracker.get_landmark(self.tracker.PINKY_TIP)
        
        if index_tip is not None and pinky_tip is not None:
            # Index and pinky should be reasonably spread for rock sign
            finger_spread = ((index_tip[0] - pinky_tip[0])**2 + (index_tip[1] - pinky_tip[1])**2)**0.5
            good_spread = finger_spread > 0.08  # Minimum spread
        else:
            good_spread = True
        
        return basic_check and thumb_tucked and good_spread
    
    def _is_call_me(self, fingers: dict, curled: dict) -> bool:
        """Thumb and pinky extended, others curled."""
        return (
            fingers['thumb'] and
            fingers['pinky'] and
            curled['index'] and
            curled['middle'] and
            curled['ring']
        )
    
    def _is_pinch_middle(self, fingers: dict, distances: dict) -> bool:
        """Thumb and middle finger close together."""
        if 'thumb_middle' not in distances:
            return False
        # Make sure it's not an OK sign (thumb_index should be further)
        thumb_index = distances.get('thumb_index', 1.0)
        return (
            distances['thumb_middle'] < self.PINCH_THRESHOLD and
            thumb_index > distances['thumb_middle']
        )
    
    def _is_peace_sign(self, fingers: dict, curled: dict) -> bool:
        """Index and middle extended, ring and pinky curled."""
        return (
            fingers['index'] and
            fingers['middle'] and
            curled['ring'] and
            curled['pinky']
        )
    
    def _is_index_up(self, fingers: dict) -> bool:
        """Index extended (used for cursor control)."""
        return (
            fingers['index'] and
            not fingers['middle'] and
            not fingers['ring'] and
            not fingers['pinky']
        )
    
    def _is_ring_curl(self, fingers: dict, curled: dict) -> bool:
        """Only ring finger curled, others extended."""
        return (
            fingers['thumb'] and
            fingers['index'] and
            fingers['middle'] and
            curled['ring'] and
            fingers['pinky']
        )
    
    def _is_middle_curl(self, fingers: dict, curled: dict) -> bool:
        """Only middle finger curled, others extended."""
        return (
            fingers['thumb'] and
            fingers['index'] and
            curled['middle'] and
            fingers['ring'] and
            fingers['pinky']
        )
    
    def _is_pinky_curl(self, fingers: dict, curled: dict) -> bool:
        """Only pinky curled, others extended."""
        return (
            fingers['thumb'] and
            fingers['index'] and
            fingers['middle'] and
            fingers['ring'] and
            curled['pinky']
        )
    
    def _is_open_palm(self, fingers: dict) -> bool:
        """All 5 fingers extended."""
        return all(fingers.values())
    
    def _is_grab(self, fingers: dict, curled: dict) -> bool:
        """Closed or semi-closed hand (merged fist and grab)."""
        # At least 3 fingers curled
        curled_count = sum([curled['index'], curled['middle'], curled['ring'], curled['pinky']])
        return curled_count >= 3
    
    def _update_state(self, gesture: Gesture) -> Gesture:
        """Update gesture state for stability tracking."""
        if self.current_state is None or self.current_state.gesture != gesture:
            self.current_state = GestureState(
                gesture=gesture,
                start_time=time.time(),
                confidence=1.0,
                stable_frames=1
            )
        else:
            self.current_state.stable_frames += 1
        
        self.frame_count += 1
        return gesture
    
    def is_gesture_stable(self, min_frames: Optional[int] = None) -> bool:
        """Check if current gesture is stable (frame-based)."""
        if self.current_state is None:
            return False
        min_frames = min_frames or self.STABILITY_FRAMES
        return self.current_state.stable_frames >= min_frames
    
    def is_gesture_held_for(self, min_duration: Optional[float] = None) -> bool:
        """Check if current gesture has been held for minimum duration (time-based).
        
        Args:
            min_duration: Minimum time in seconds the gesture must be held.
                         If None, uses the instance's min_gesture_duration.
        
        Returns:
            True if gesture has been held for at least min_duration seconds.
        """
        if self.current_state is None:
            return False
        min_duration = min_duration if min_duration is not None else self.min_gesture_duration
        return self.current_state.duration() >= min_duration
    
    def is_gesture_ready(self, min_frames: Optional[int] = None, min_duration: Optional[float] = None) -> bool:
        """Check if gesture is stable AND held for minimum duration.
        
        This combines both frame-based and time-based checks to prevent
        false triggers from momentary gestures.
        
        Args:
            min_frames: Minimum consecutive frames (default: STABILITY_FRAMES)
            min_duration: Minimum duration in seconds (default: min_gesture_duration)
        
        Returns:
            True if both stability and duration requirements are met.
        """
        return self.is_gesture_stable(min_frames) and self.is_gesture_held_for(min_duration)
    
    def get_gesture_duration(self) -> float:
        """Get how long current gesture has been held."""
        if self.current_state is None:
            return 0.0
        return self.current_state.duration()
