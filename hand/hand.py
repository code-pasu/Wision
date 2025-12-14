"""
Hand Gesture Control System - Entry Point

This module provides a hand gesture control system using MediaPipe.
Uses only reliable gestures with OK Sign as mode switcher.

Run with: python hand/hand.py
Or:       python -m hand

Modules:
    - gestures.py: Gesture and mode enums/definitions
    - tracker.py: Hand tracking with MediaPipe
    - recognizer.py: Gesture recognition logic
    - actions.py: System control actions
    - smoothing.py: Cursor smoothing filters
    - controller.py: Main controller class

Gesture Mappings:
    Mode Switch:
        - OK Sign (hold 5 frames): Cycle through modes

    CURSOR Mode:
        - Pointing: Move cursor
        - Pinch Middle/L Sign: Left click
        - Rock Sign: Right click
        - Call Me: Double click
        - Peace Sign: Scroll (angle-based)

    SCROLL Mode:
        - Peace Sign: Scroll up/down based on tilt angle
        - Pointing: Move cursor

    WINDOW Mode:
        - Open Palm: Maximize window
        - Fist: Minimize window
        - Rock Sign: Switch window (Alt+Tab)
        - Pinch Middle: Show desktop
        - Call Me: Close window (Alt+F4)
        - Peace Sign: Screenshot

    MEDIA Mode:
        - Open Palm: Play/Pause
        - Pointing: Next track
        - Peace Sign: Previous track
        - Pinch Middle: Volume up
        - Rock Sign: Volume down
        - Fist: Mute
"""

from .controller import HandGestureController, main

if __name__ == '__main__':
    main()
