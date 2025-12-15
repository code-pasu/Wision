"""
Gesture recognition from hand tracking data.

This module provides the GestureRecognizer class which:
- Analyzes finger states, joint angles, and distances to classify gestures
- Uses priority-ordered detection to avoid misclassification
- Implements multi-layer validation (basic + geometric + temporal)
- Tracks gesture stability (consecutive frames) and duration (time-based)

Detection Pipeline:
1. Basic finger state check (extended/curled)
2. Geometric validation (angles, distances, positions)
3. Frame stability check (N consecutive frames)
4. Time duration check (minimum hold time)

Gestures are checked from most specific to most general to prevent
ambiguous classifications.
"""

import time
import numpy as np
from typing import Optional
from .tracker import HandTracker
from .gestures import Gesture, GestureState


class GestureRecognizer:
    """
    Recognizes hand gestures from tracking data with robust multi-layer validation.
    
    Features:
    - Priority-ordered gesture detection (specific → general)
    - Geometric validation using joint angles and landmark distances
    - Frame-based stability tracking
    - Time-based duration thresholds to prevent false triggers
    
    Attributes:
        tracker: HandTracker instance providing landmark data
        current_state: Current GestureState for stability/duration tracking
        min_gesture_duration: Minimum seconds a gesture must be held
    """
    
    # Distance thresholds (normalized to hand size)
    OK_SIGN_THRESHOLD = 0.05      # Max thumb-index distance for OK sign
    PINCH_THRESHOLD = 0.06        # Max distance for pinch gestures
    L_SIGN_THUMB_EXTENSION = 0.12 # Min thumb-to-index-MCP distance for L sign
    L_SIGN_WRIST_DISTANCE = 0.15  # Min thumb-to-wrist distance for L sign
    ROCK_THUMB_MAX_DIST = 0.18    # Max thumb-to-middle-MCP for rock (thumb not extended far)
    ROCK_FINGER_SPREAD = 0.06     # Min index-pinky spread for rock sign (relaxed)
    INDEX_UP_TIP_OFFSET = 0.02    # Min vertical offset for index pointing up
    
    # Angle thresholds (degrees)
    L_SIGN_THUMB_ANGLE = 155      # Min thumb IP joint angle for L sign
    INDEX_UP_JOINT_ANGLE = 140    # Min joint angle for extended index
    ROCK_EXTENDED_ANGLE = 120     # Min joint angle for rock sign extended fingers (relaxed)
    
    # Stability thresholds
    STABILITY_FRAMES = 3          # Min consecutive frames for gesture stability
    MIN_GESTURE_DURATION = 0.15   # Default min seconds for gesture activation
    
    def __init__(self, tracker: HandTracker, min_gesture_duration: Optional[float] = None):
        """
        Initialize the gesture recognizer.
        
        Args:
            tracker: HandTracker instance for landmark data
            min_gesture_duration: Optional custom duration threshold (seconds)
        """
        self.tracker = tracker
        self.current_state: Optional[GestureState] = None
        self.frame_count = 0
        self.min_gesture_duration = min_gesture_duration or self.MIN_GESTURE_DURATION
        
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
        """L Sign: thumb + index fully extended, forming an L shape.
        
        Robust detection with strict thumb extension validation:
        1. Basic check: thumb and index extended, others curled
        2. Thumb angle: IP joint must be very straight (> 155°)
        3. Thumb extension: tip must be far from index MCP (> 0.12)
        4. Thumb position: tip must be away from wrist (> 0.15)
        
        This strict validation prevents false triggers when:
        - Thumb is only partially extended
        - Thumb is resting near the palm
        - Transitioning from other gestures
        """
        if self.tracker.landmarks is None:
            return False
        
        # Layer 1: Basic finger state checks
        basic_check = (
            fingers['thumb'] and
            fingers['index'] and
            curled['middle'] and
            curled['ring'] and
            curled['pinky']
        )
        
        if not basic_check:
            return False
        
        # Layer 2: Get thumb landmarks for geometric validation
        thumb_tip = self.tracker.get_landmark(self.tracker.THUMB_TIP)
        thumb_ip = self.tracker.get_landmark(self.tracker.THUMB_IP)
        thumb_mcp = self.tracker.get_landmark(self.tracker.THUMB_MCP)
        index_mcp = self.tracker.get_landmark(self.tracker.INDEX_MCP)
        wrist = self.tracker.get_landmark(self.tracker.WRIST)
        
        if None in (thumb_tip, thumb_ip, thumb_mcp, index_mcp, wrist):
            return basic_check
        
        # Layer 3: Thumb angle at IP joint - must be very straight
        thumb_angle = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        if thumb_angle < self.L_SIGN_THUMB_ANGLE:
            return False
        
        # Layer 4: Thumb tip must be far from index MCP (extended outward)
        thumb_to_index_mcp = (
            (thumb_tip[0] - index_mcp[0])**2 +  # type: ignore
            (thumb_tip[1] - index_mcp[1])**2  # type: ignore
        )**0.5
        
        if thumb_to_index_mcp < self.L_SIGN_THUMB_EXTENSION:
            return False
        
        # Layer 5: Thumb tip should be away from wrist (not curled back)
        thumb_to_wrist = (
            (thumb_tip[0] - wrist[0])**2 +  # type: ignore
            (thumb_tip[1] - wrist[1])**2  # type: ignore
        )**0.5
        
        if thumb_to_wrist < self.L_SIGN_WRIST_DISTANCE:
            return False
        
        return True
    
    def _calculate_angle(self, p1, p2, p3) -> float:
        """Calculate angle at p2 between vectors p1-p2 and p2-p3.
        
        Uses 2D projection for stability (ignores Z depth).
        
        Args:
            p1, p2, p3: Landmark coordinates (x, y, z)
            
        Returns:
            Angle in degrees at p2 (0-180)
        """
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
        """Rock sign: index + pinky extended, middle + ring curled, thumb tucked.
        
        Improved detection with balanced validation:
        1. Basic check: index and pinky must be extended, middle and ring curled
        2. Thumb check: must NOT be fully extended (key differentiator from L sign)
        3. Finger spread: index and pinky should be spread apart
        4. Joint angles: extended fingers should be reasonably straight
        
        This prevents confusion with:
        - L sign (which has thumb extended outward)
        - Peace sign (which has middle extended)
        - Call me (which has thumb extended to the side)
        
        Note: We focus on thumb NOT being extended rather than requiring it to
        be very close to palm, as natural rock sign hand position varies.
        """
        if self.tracker.landmarks is None:
            return False
        
        # Layer 1: Core finger state checks - index and pinky up, middle and ring down
        if not fingers['index'] or not fingers['pinky']:
            return False
        if not curled['middle'] or not curled['ring']:
            return False
        
        # Layer 2: Thumb must NOT be extended (key differentiator from L sign)
        # We check this explicitly rather than relying only on fingers['thumb']
        thumb_tip = self.tracker.get_landmark(self.tracker.THUMB_TIP)
        thumb_ip = self.tracker.get_landmark(self.tracker.THUMB_IP)
        thumb_mcp = self.tracker.get_landmark(self.tracker.THUMB_MCP)
        middle_mcp = self.tracker.get_landmark(self.tracker.MIDDLE_MCP)
        index_mcp = self.tracker.get_landmark(self.tracker.INDEX_MCP)
        wrist = self.tracker.get_landmark(self.tracker.WRIST)
        
        if thumb_tip is not None and middle_mcp is not None:
            # Check thumb tip is not too far from middle MCP (i.e., not extended outward)
            thumb_to_middle = (
                (thumb_tip[0] - middle_mcp[0])**2 + 
                (thumb_tip[1] - middle_mcp[1])**2
            )**0.5
            
            # If thumb is extended far from palm center, it's likely L sign or call me
            if thumb_to_middle > self.ROCK_THUMB_MAX_DIST:
                return False
        
        # Also verify thumb is not in L sign position (thumb extended to side with straight angle)
        if thumb_tip is not None and thumb_ip is not None and thumb_mcp is not None:
            thumb_angle = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
            if thumb_angle > 160:  # Very straight thumb = likely L sign
                # Additional check: is thumb pointing away from hand?
                if wrist is not None and index_mcp is not None:
                    thumb_to_wrist = (
                        (thumb_tip[0] - wrist[0])**2 + 
                        (thumb_tip[1] - wrist[1])**2
                    )**0.5
                    thumb_to_index = (
                        (thumb_tip[0] - index_mcp[0])**2 + 
                        (thumb_tip[1] - index_mcp[1])**2
                    )**0.5
                    # If thumb is far from wrist and index, it's extended outward
                    if thumb_to_wrist > 0.15 and thumb_to_index > 0.12:
                        return False
        
        # Layer 3: Check that index and pinky tips are spread apart
        index_tip = self.tracker.get_landmark(self.tracker.INDEX_TIP)
        pinky_tip = self.tracker.get_landmark(self.tracker.PINKY_TIP)
        
        if index_tip is not None and pinky_tip is not None:
            finger_spread = (
                (index_tip[0] - pinky_tip[0])**2 + 
                (index_tip[1] - pinky_tip[1])**2
            )**0.5
            if finger_spread < self.ROCK_FINGER_SPREAD:
                return False
        
        # Layer 4: Verify index and pinky are reasonably extended (relaxed angle check)
        index_pip = self.tracker.get_landmark(self.tracker.INDEX_PIP)
        index_dip = self.tracker.get_landmark(self.tracker.INDEX_DIP)
        
        if index_tip is not None and index_pip is not None and index_dip is not None and index_mcp is not None:
            index_pip_angle = self._calculate_angle(index_mcp, index_pip, index_dip)
            if index_pip_angle < self.ROCK_EXTENDED_ANGLE:  # Relaxed from 130 to 120
                return False
        
        return True
    
    def _is_call_me(self, fingers: dict, curled: dict) -> bool:
        """Call me sign: thumb + pinky extended, others curled.
        
        Detection requires:
        - Thumb clearly extended outward
        - Pinky clearly extended
        - Index, middle, ring all curled
        """
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
        """Index and middle extended, ring and pinky curled.
        
        Validation:
        - Index and middle fingers clearly extended
        - Ring and pinky fingers clearly curled
        - Thumb state ignored (can be in any position)
        """
        return (
            fingers['index'] and
            fingers['middle'] and
            curled['ring'] and
            curled['pinky']
        )
    
    def _is_index_up(self, fingers: dict) -> bool:
        """Index finger extended for cursor control.
        
        Robust detection with geometric validation:
        - Index finger must be clearly extended (straight joints)
        - Index fingertip should be above index MCP (pointing upward)
        - Middle, ring, and pinky must be curled
        - Thumb state is ignored (allows natural hand position)
        
        This prevents false triggers from partially extended fingers
        or when transitioning between gestures.
        """
        if self.tracker.landmarks is None:
            return False
        
        # Basic finger state check
        basic_check = (
            fingers['index'] and
            not fingers['middle'] and
            not fingers['ring'] and
            not fingers['pinky']
        )
        
        if not basic_check:
            return False
        
        # Geometric validation: index tip should be above index MCP
        index_tip = self.tracker.get_landmark(self.tracker.INDEX_TIP)
        index_mcp = self.tracker.get_landmark(self.tracker.INDEX_MCP)
        index_pip = self.tracker.get_landmark(self.tracker.INDEX_PIP)
        index_dip = self.tracker.get_landmark(self.tracker.INDEX_DIP)
        
        if None in (index_tip, index_mcp, index_pip, index_dip):
            return basic_check
        
        # Check that index finger joints are relatively straight
        pip_angle = self._calculate_angle(index_mcp, index_pip, index_dip)
        dip_angle = self._calculate_angle(index_pip, index_dip, index_tip)
        
        # Both angles should be > 140 degrees for extended finger
        if pip_angle < self.INDEX_UP_JOINT_ANGLE or dip_angle < self.INDEX_UP_JOINT_ANGLE:
            return False
        
        # Index tip should be above (lower Y value) or at same level as MCP
        # Allow small tolerance for horizontal pointing
        tip_above_mcp = index_tip[1] <= index_mcp[1] + self.INDEX_UP_TIP_OFFSET  # type: ignore
        
        # Additional: check that index tip is away from palm (finger extended outward)
        wrist = self.tracker.get_landmark(self.tracker.WRIST)
        if wrist is not None:
            tip_to_wrist = ((index_tip[0] - wrist[0])**2 + (index_tip[1] - wrist[1])**2)**0.5  # type: ignore
            # Index tip should be reasonably far from wrist
            if tip_to_wrist < 0.15:
                return False
        
        return tip_above_mcp
    
    def _is_ring_curl(self, fingers: dict, curled: dict) -> bool:
        """Ring curl: only ring finger curled, all others extended.
        
        Used for middle-click in cursor mode.
        """
        return (
            fingers['thumb'] and
            fingers['index'] and
            fingers['middle'] and
            curled['ring'] and
            fingers['pinky']
        )
    
    def _is_middle_curl(self, fingers: dict, curled: dict) -> bool:
        """Middle curl: only middle finger curled, all others extended.
        
        Used for scroll-click in cursor mode.
        """
        return (
            fingers['thumb'] and
            fingers['index'] and
            curled['middle'] and
            fingers['ring'] and
            fingers['pinky']
        )
    
    def _is_pinky_curl(self, fingers: dict, curled: dict) -> bool:
        """Pinky curl: only pinky finger curled, all others extended."""
        return (
            fingers['thumb'] and
            fingers['index'] and
            fingers['middle'] and
            fingers['ring'] and
            curled['pinky']
        )
    
    def _is_open_palm(self, fingers: dict) -> bool:
        """Open palm: all 5 fingers fully extended.
        
        Used for maximize (window mode) or play/pause (media mode).
        """
        return all(fingers.values())
    
    def _is_grab(self, fingers: dict, curled: dict) -> bool:
        """Grab/Fist: closed or semi-closed hand.
        
        Detection: at least 3 of 4 fingers (excluding thumb) are curled.
        Used for drag (cursor), minimize (window), or mute (media).
        """
        curled_count = sum([curled['index'], curled['middle'], curled['ring'], curled['pinky']])
        return curled_count >= 3
    
    def _update_state(self, gesture: Gesture) -> Gesture:
        """Update gesture state for stability and duration tracking.
        
        Creates a new GestureState when gesture changes, or increments
        stable_frames when the same gesture persists.
        """
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
        """Check if current gesture is stable (frame-based).
        
        Args:
            min_frames: Minimum consecutive frames required (default: STABILITY_FRAMES)
            
        Returns:
            True if gesture has been detected for at least min_frames consecutive frames.
        """
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
        false triggers from momentary gestures. Both conditions must be met.
        
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
