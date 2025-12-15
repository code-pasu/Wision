"""
Wision -- Hand Gesture Controller Launcher

This script provides a quick way to start the gesture controller.
For testing gestures without triggering actions, use test_gestures.py instead.

Usage:
    python run_hand.py

Controls:
    OK Sign (hold) -- Switch between control modes
    Q or ESC       -- Quit the application
""" 

from hand.hand import HandGestureController, main

if __name__ == '__main__':
    main()
 