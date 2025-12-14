"""
Gesture and control mode definitions.

This module defines:
- ControlMode: The 4 operating modes (CURSOR, SCROLL, WINDOW, MEDIA)
- Gesture: All recognized hand gestures
- GestureState: Tracks gesture timing and stability
- GESTURE_DESCRIPTIONS: Human-readable gesture descriptions for UI
"""

from enum import Enum, auto
from dataclasses import dataclass
import time


class ControlMode(Enum):
    """Control modes switched by OK sign."""
    CURSOR = auto()
    SCROLL = auto()
    WINDOW = auto()
    MEDIA = auto()


class Gesture(Enum):
    """Recognized hand gestures."""
    NONE = auto()
    OK_SIGN = auto()         # Mode switcher (thumb+index circle, others extended)
    OPEN_PALM = auto()       # All fingers extended
    GRAB = auto()            # Closed/semi-closed hand (merged fist+grab)
    PINCH_MIDDLE = auto()    # Thumb + middle touch
    ROCK_SIGN = auto()       # Index + pinky extended, middle+ring curled, NO thumb
    CALL_ME = auto()         # Thumb + pinky extended
    PEACE_SIGN = auto()      # Index + middle extended (V sign)
    INDEX_UP = auto()        # Index extended only (cursor move)
    L_SIGN = auto()          # Thumb + index extended (L shape) - cursor + click
    RING_CURL = auto()       # Only ring finger curled, others extended
    MIDDLE_CURL = auto()     # Only middle finger curled, others extended
    PINKY_CURL = auto()      # Only pinky curled, others extended


@dataclass
class GestureState:
    """Tracks gesture timing and stability."""
    gesture: Gesture
    start_time: float
    confidence: float
    stable_frames: int
    
    def duration(self) -> float:
        return time.time() - self.start_time


# Gesture descriptions for UI
GESTURE_DESCRIPTIONS = {
    Gesture.NONE: "No gesture detected",
    Gesture.OK_SIGN: "Thumb + Index circle, others open",
    Gesture.OPEN_PALM: "All 5 fingers extended",
    Gesture.GRAB: "Closed or semi-closed hand",
    Gesture.PINCH_MIDDLE: "Thumb + Middle finger touching",
    Gesture.ROCK_SIGN: "Index + Pinky extended (no thumb)",
    Gesture.CALL_ME: "Thumb + Pinky extended",
    Gesture.PEACE_SIGN: "Index + Middle extended (V)",
    Gesture.INDEX_UP: "Only index extended (cursor move)",
    Gesture.L_SIGN: "Thumb + Index (L shape = click)",
    Gesture.RING_CURL: "Only ring finger curled",
    Gesture.MIDDLE_CURL: "Only middle finger curled",
    Gesture.PINKY_CURL: "Only pinky finger curled",
}
