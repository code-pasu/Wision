"""
Wision -- Hand gesture recognition and control package.

This package provides real-time hand gesture control for Windows.
Uses MediaPipe for tracking, custom rules for recognition, and
PyAutoGUI for system control.

Quick start:
    from hand import HandGestureController
    controller = HandGestureController()
    controller.run()
"""

from .gestures import Gesture, ControlMode, GestureState, GESTURE_DESCRIPTIONS
from .tracker import HandTracker
from .recognizer import GestureRecognizer
from .actions import ActionController, ActionType
from .smoothing import OneEuroFilter, AdaptiveSmoother
from .controller import HandGestureController, main

__all__ = [
    'Gesture',
    'ControlMode', 
    'GestureState',
    'GESTURE_DESCRIPTIONS',
    'HandTracker',
    'GestureRecognizer',
    'ActionController',
    'ActionType',
    'OneEuroFilter',
    'AdaptiveSmoother',
    'HandGestureController',
    'main',
]
