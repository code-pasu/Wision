"""
Gesture and control mode definitions.

This module defines the core data structures for gesture recognition:
- ControlMode: The 4 operating modes (CURSOR, SCROLL, WINDOW, MEDIA)
- Gesture: All 12 recognized hand gestures
- GestureState: Tracks gesture timing, stability, and confidence
- GESTURE_DESCRIPTIONS: Human-readable descriptions for UI display

The gesture system uses a hierarchical approach:
1. Raw detection (finger states)
2. Geometric validation (angles, distances)
3. Temporal validation (frames, duration)
"""

from enum import Enum, auto
from dataclasses import dataclass
import time


class ControlMode(Enum):
    """
    Control modes switched by OK sign gesture.
    
    Each mode maps gestures to different system actions:
    - CURSOR: Mouse control (move, click, drag, scroll)
    - SCROLL: Dedicated scrolling with angle-based direction
    - WINDOW: Window management (maximize, minimize, switch, close)
    - MEDIA: Media playback control (play/pause, volume, track navigation)
    """
    CURSOR = auto()
    SCROLL = auto()
    WINDOW = auto()
    MEDIA = auto()


class Gesture(Enum):
    """
    Recognized hand gestures.
    
    Detection priority (checked in this order to avoid misclassification):
    1. OK_SIGN - Most specific (thumb-index circle)
    2. CALL_ME - Thumb + pinky only
    3. L_SIGN - Thumb + index extended (strict thumb check)
    4. ROCK_SIGN - Index + pinky, thumb tucked
    5. PINCH_MIDDLE - Thumb-middle distance
    6. PEACE_SIGN - Index + middle extended
    7. Single curl gestures (RING_CURL, MIDDLE_CURL, PINKY_CURL)
    8. OPEN_PALM - All fingers extended
    9. INDEX_UP - Index only (for cursor movement)
    10. GRAB - Closed hand (most general)
    """
    NONE = auto()
    OK_SIGN = auto()         # Mode switcher (thumb+index circle, others extended)
    OPEN_PALM = auto()       # All fingers extended
    GRAB = auto()            # Closed/semi-closed hand (3+ fingers curled)
    PINCH_MIDDLE = auto()    # Thumb + middle finger touching
    ROCK_SIGN = auto()       # Index + pinky extended, thumb tucked, middle+ring curled
    CALL_ME = auto()         # Thumb + pinky extended, others curled
    PEACE_SIGN = auto()      # Index + middle extended (V sign)
    INDEX_UP = auto()        # Index extended only (cursor movement)
    L_SIGN = auto()          # Thumb + index fully extended (L shape) - cursor + click
    RING_CURL = auto()       # Only ring finger curled, others extended
    MIDDLE_CURL = auto()     # Only middle finger curled, others extended
    PINKY_CURL = auto()      # Only pinky curled, others extended


@dataclass
class GestureState:
    """
    Tracks gesture timing and stability for action triggering.
    
    Attributes:
        gesture: The detected Gesture enum value
        start_time: Unix timestamp when gesture was first detected
        confidence: Detection confidence (0.0-1.0, reserved for future ML use)
        stable_frames: Number of consecutive frames with same gesture
        
    Methods:
        duration(): Returns seconds since gesture started
    """
    gesture: Gesture
    start_time: float
    confidence: float
    stable_frames: int
    
    def duration(self) -> float:
        """Get how long this gesture has been held in seconds."""
        return time.time() - self.start_time


# Human-readable gesture descriptions for UI display
GESTURE_DESCRIPTIONS = {
    Gesture.NONE: "No gesture detected",
    Gesture.OK_SIGN: "Thumb + Index circle, others open",
    Gesture.OPEN_PALM: "All 5 fingers extended",
    Gesture.GRAB: "Closed or semi-closed hand",
    Gesture.PINCH_MIDDLE: "Thumb + Middle finger touching",
    Gesture.ROCK_SIGN: "Index + Pinky extended, thumb tucked",
    Gesture.CALL_ME: "Thumb + Pinky extended",
    Gesture.PEACE_SIGN: "Index + Middle extended (V)",
    Gesture.INDEX_UP: "Only index extended (cursor move)",
    Gesture.L_SIGN: "Thumb + Index fully extended (L shape)",
    Gesture.RING_CURL: "Only ring finger curled",
    Gesture.MIDDLE_CURL: "Only middle finger curled",
    Gesture.PINKY_CURL: "Only pinky finger curled",
}
