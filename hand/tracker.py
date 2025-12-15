"""
Hand tracking using Google MediaPipe.

This module provides the HandTracker class which:
- Captures 21 hand landmarks from video frames using MediaPipe
- Computes finger extension/curl states based on joint angles
- Calculates fingertip distances for pinch/OK sign detection
- Measures peace sign tilt angle for scroll direction control

Landmark Indices (MediaPipe hand model):
    0: WRIST
    1-4: THUMB (CMC, MCP, IP, TIP)
    5-8: INDEX (MCP, PIP, DIP, TIP)
    9-12: MIDDLE (MCP, PIP, DIP, TIP)
    13-16: RING (MCP, PIP, DIP, TIP)
    17-20: PINKY (MCP, PIP, DIP, TIP)

Finger State Detection:
- Extended: Joint angles > 140° (relatively straight)
- Curled: PIP joint angle < 150° (bent)
- Thumb uses special logic (angle + distance from palm)

Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict


class HandTracker:
    """
    Tracks hand landmarks using Google MediaPipe.
    
    This class wraps MediaPipe's hand tracking solution and provides
    convenient methods for gesture recognition:
    - Landmark access (normalized and pixel coordinates)
    - Finger extension/curl state detection
    - Fingertip distance calculations
    - Peace sign angle measurement
    
    Attributes:
        landmarks: Current frame's 21 hand landmarks (or None)
        handedness: 'Left' or 'Right' (or None)
    """
    
    # Landmark indices (MediaPipe convention)
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(
        self,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
        max_hands: int = 1
    ):
        """
        Initialize the hand tracker.
        
        Args:
            detection_confidence: Min confidence for initial hand detection (0-1)
            tracking_confidence: Min confidence for landmark tracking (0-1)
            max_hands: Maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands # pyright: ignore[reportAttributeAccessIssue]
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils # pyright: ignore[reportAttributeAccessIssue]
        self.landmarks: Optional[List] = None
        self.handedness: Optional[str] = None
        
    def process(self, frame: np.ndarray) -> bool:
        """Process frame and detect hands."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            self.landmarks = results.multi_hand_landmarks[0].landmark
            if results.multi_handedness:
                self.handedness = results.multi_handedness[0].classification[0].label
            return True
        
        self.landmarks = None
        self.handedness = None
        return False
    
    def get_landmark(self, idx: int) -> Optional[Tuple[float, float, float]]:
        """Get normalized coordinates of a landmark."""
        if self.landmarks is None:
            return None
        lm = self.landmarks[idx]
        return (lm.x, lm.y, lm.z)
    
    def get_landmark_pixel(
        self, 
        idx: int, 
        frame_width: int, 
        frame_height: int
    ) -> Optional[Tuple[int, int]]:
        """Get pixel coordinates of a landmark."""
        lm = self.get_landmark(idx)
        if lm is None:
            return None
        return (int(lm[0] * frame_width), int(lm[1] * frame_height))
    
    def _calculate_angle(
        self, 
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float]
    ) -> float:
        """Calculate angle at p2 between p1-p2-p3."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 180.0
            
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _is_finger_extended(self, mcp: int, pip: int, dip: int, tip: int) -> bool:
        """Check if finger is extended using joint angles."""
        if self.landmarks is None:
            return False
            
        # Get joint positions
        mcp_pos = self.get_landmark(mcp)
        pip_pos = self.get_landmark(pip)
        dip_pos = self.get_landmark(dip)
        tip_pos = self.get_landmark(tip)
        
        if None in (mcp_pos, pip_pos, dip_pos, tip_pos):
            return False
        
        # Calculate angles at PIP and DIP joints
        pip_angle = self._calculate_angle(mcp_pos, pip_pos, dip_pos) # type: ignore
        dip_angle = self._calculate_angle(pip_pos, dip_pos, tip_pos) # type: ignore
        
        # Extended if both angles are relatively straight (> 140 degrees)
        return pip_angle > 140 and dip_angle > 140
    
    def _is_finger_curled(self, mcp: int, pip: int, dip: int, tip: int) -> bool:
        """Check if finger is curled using joint angles."""
        if self.landmarks is None:
            return False
            
        mcp_pos = self.get_landmark(mcp)
        pip_pos = self.get_landmark(pip)
        dip_pos = self.get_landmark(dip)
        
        if None in (mcp_pos, pip_pos, dip_pos):
            return False
        
        pip_angle = self._calculate_angle(mcp_pos, pip_pos, dip_pos) # type: ignore
        
        # Curled if PIP angle is less than 150 degrees
        return pip_angle < 150
    
    def is_thumb_extended(self) -> bool:
        """Check if thumb is extended."""
        if self.landmarks is None:
            return False
        
        thumb_tip = self.get_landmark(self.THUMB_TIP)
        thumb_ip = self.get_landmark(self.THUMB_IP)
        thumb_mcp = self.get_landmark(self.THUMB_MCP)
        index_mcp = self.get_landmark(self.INDEX_MCP)
        
        if None in (thumb_tip, thumb_ip, thumb_mcp, index_mcp):
            return False
        
        # Calculate thumb angle
        angle = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip) # type: ignore
        
        # Also check if thumb is away from palm
        thumb_to_index = np.sqrt(
            (thumb_tip[0] - index_mcp[0])**2 +  # type: ignore
            (thumb_tip[1] - index_mcp[1])**2 # type: ignore
        )
        
        return angle > 140 and thumb_to_index > 0.08
    
    def is_index_extended(self) -> bool:
        """Check if index finger is extended."""
        return self._is_finger_extended(
            self.INDEX_MCP, self.INDEX_PIP, 
            self.INDEX_DIP, self.INDEX_TIP
        )
    
    def is_middle_extended(self) -> bool:
        """Check if middle finger is extended."""
        return self._is_finger_extended(
            self.MIDDLE_MCP, self.MIDDLE_PIP,
            self.MIDDLE_DIP, self.MIDDLE_TIP
        )
    
    def is_ring_extended(self) -> bool:
        """Check if ring finger is extended."""
        return self._is_finger_extended(
            self.RING_MCP, self.RING_PIP,
            self.RING_DIP, self.RING_TIP
        )
    
    def is_pinky_extended(self) -> bool:
        """Check if pinky finger is extended."""
        return self._is_finger_extended(
            self.PINKY_MCP, self.PINKY_PIP,
            self.PINKY_DIP, self.PINKY_TIP
        )
    
    def is_index_curled(self) -> bool:
        """Check if index finger is curled."""
        return self._is_finger_curled(
            self.INDEX_MCP, self.INDEX_PIP,
            self.INDEX_DIP, self.INDEX_TIP
        )
    
    def is_middle_curled(self) -> bool:
        """Check if middle finger is curled."""
        return self._is_finger_curled(
            self.MIDDLE_MCP, self.MIDDLE_PIP,
            self.MIDDLE_DIP, self.MIDDLE_TIP
        )
    
    def is_ring_curled(self) -> bool:
        """Check if ring finger is curled."""
        return self._is_finger_curled(
            self.RING_MCP, self.RING_PIP,
            self.RING_DIP, self.RING_TIP
        )
    
    def is_pinky_curled(self) -> bool:
        """Check if pinky finger is curled."""
        return self._is_finger_curled(
            self.PINKY_MCP, self.PINKY_PIP,
            self.PINKY_DIP, self.PINKY_TIP
        )
    
    def get_finger_states(self) -> Dict[str, bool]:
        """Get extension state of all fingers."""
        return {
            'thumb': self.is_thumb_extended(),
            'index': self.is_index_extended(),
            'middle': self.is_middle_extended(),
            'ring': self.is_ring_extended(),
            'pinky': self.is_pinky_extended()
        }
    
    def get_finger_curl_states(self) -> Dict[str, bool]:
        """Get curl state of all fingers (except thumb)."""
        return {
            'index': self.is_index_curled(),
            'middle': self.is_middle_curled(),
            'ring': self.is_ring_curled(),
            'pinky': self.is_pinky_curled()
        }
    
    def get_finger_distances(self) -> Dict[str, float]:
        """Get distances between fingertips and thumb."""
        if self.landmarks is None:
            return {}
        
        thumb_tip = self.get_landmark(self.THUMB_TIP)
        if thumb_tip is None:
            return {}
        
        distances = {}
        tips = [
            ('thumb_index', self.INDEX_TIP),
            ('thumb_middle', self.MIDDLE_TIP),
            ('thumb_ring', self.RING_TIP),
            ('thumb_pinky', self.PINKY_TIP)
        ]
        
        for name, tip_idx in tips:
            tip = self.get_landmark(tip_idx)
            if tip:
                dist = np.sqrt(
                    (thumb_tip[0] - tip[0])**2 +
                    (thumb_tip[1] - tip[1])**2 +
                    (thumb_tip[2] - tip[2])**2
                )
                distances[name] = dist
        
        return distances
    
    def get_peace_sign_angle(self) -> Optional[float]:
        """Get the angle of the peace sign (V) relative to vertical."""
        if self.landmarks is None:
            return None
        
        index_tip = self.get_landmark(self.INDEX_TIP)
        middle_tip = self.get_landmark(self.MIDDLE_TIP)
        index_mcp = self.get_landmark(self.INDEX_MCP)
        middle_mcp = self.get_landmark(self.MIDDLE_MCP)
        
        if None in (index_tip, middle_tip, index_mcp, middle_mcp):
            return None
        
        # Calculate midpoint of fingertips
        tip_mid_x = (index_tip[0] + middle_tip[0]) / 2 # type: ignore
        tip_mid_y = (index_tip[1] + middle_tip[1]) / 2 # type: ignore
        
        # Calculate midpoint of MCPs
        mcp_mid_x = (index_mcp[0] + middle_mcp[0]) / 2 # type: ignore
        mcp_mid_y = (index_mcp[1] + middle_mcp[1]) / 2 # type: ignore
        
        # Calculate angle from MCP midpoint to tip midpoint
        dx = tip_mid_x - mcp_mid_x
        dy = tip_mid_y - mcp_mid_y  # Note: y increases downward
        
        # Angle relative to vertical (0 = pointing up, positive = tilted right)
        angle = np.degrees(np.arctan2(dx, -dy))
        
        return angle
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on frame."""
        if self.landmarks is None:
            return frame
        
        # Create a proper NormalizedLandmarkList for drawing
        class LandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks
        
        landmark_list = LandmarkList(self.landmarks)
        
        self.mp_draw.draw_landmarks(
            frame,
            landmark_list,
            self.mp_hands.HAND_CONNECTIONS
        )
        
        return frame
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
