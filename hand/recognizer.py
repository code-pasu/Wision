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
    
    def __init__(self, tracker: HandTracker):
        self.tracker = tracker
        self.current_state: Optional[GestureState] = None
        self.frame_count = 0
        
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
        """Thumb and index extended (L shape), others curled."""
        return (
            fingers['thumb'] and
            fingers['index'] and
            curled['middle'] and
            curled['ring'] and
            curled['pinky']
        )
    
    def _is_rock_sign(self, fingers: dict, curled: dict) -> bool:
        """Index and pinky extended, middle and ring curled, NO thumb."""
        return (
            not fingers['thumb'] and  # Thumb must NOT be extended
            fingers['index'] and 
            fingers['pinky'] and
            curled['middle'] and
            curled['ring']
        )
    
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
    
    def is_gesture_stable(self, min_frames: int = None) -> bool: # type: ignore
        """Check if current gesture is stable."""
        if self.current_state is None:
            return False
        min_frames = min_frames or self.STABILITY_FRAMES
        return self.current_state.stable_frames >= min_frames
    
    def get_gesture_duration(self) -> float:
        """Get how long current gesture has been held."""
        if self.current_state is None:
            return 0.0
        return self.current_state.duration()
