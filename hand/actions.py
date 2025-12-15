"""
Action controller for executing system commands.

This module provides the ActionController class which:
- Translates gestures into OS actions via PyAutoGUI
- Manages control mode switching (CURSOR -> SCROLL -> WINDOW -> MEDIA)
- Implements cooldowns to prevent action spam
- Provides smooth relative cursor movement

Supported actions: mouse clicks, scrolling, hotkeys, media keys.
"""

import time
import pyautogui
from enum import Enum, auto
from typing import Optional, Tuple
from .gestures import Gesture, ControlMode
from .smoothing import AdaptiveSmoother


# Configure pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class ActionType(Enum):
    """Types of actions that can be performed."""
    NONE = auto()
    MODE_SWITCH = auto()
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    MIDDLE_CLICK = auto()
    DOUBLE_CLICK = auto()
    DRAG_START = auto()
    DRAG_END = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    MAXIMIZE = auto()
    MINIMIZE = auto()
    CLOSE = auto()
    SWITCH_WINDOW = auto()
    SHOW_DESKTOP = auto()
    SCREENSHOT = auto()
    PLAY_PAUSE = auto()
    NEXT_TRACK = auto()
    PREV_TRACK = auto()
    VOLUME_UP = auto()
    VOLUME_DOWN = auto()
    MUTE = auto()


class ActionController:
    """Handles system control actions based on gestures."""
    
    # Mode rotation order
    MODE_ORDER = [
        ControlMode.CURSOR, 
        ControlMode.SCROLL,
        ControlMode.WINDOW, 
        ControlMode.MEDIA
    ]
    
    def __init__(self):
        self.current_mode = ControlMode.CURSOR
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoother = AdaptiveSmoother(self.screen_width, self.screen_height)
        
        # Action cooldowns (in seconds)
        self.cooldowns = {
            ActionType.MODE_SWITCH: 0.8,
            ActionType.LEFT_CLICK: 0.5,   # Increased to prevent accidental double clicks
            ActionType.RIGHT_CLICK: 0.7,  # Increased to prevent accidental double clicks
            ActionType.DOUBLE_CLICK: 0.6,
            ActionType.MIDDLE_CLICK: 0.5,
            ActionType.MAXIMIZE: 0.8,
            ActionType.MINIMIZE: 0.8,
            ActionType.CLOSE: 1.0,
            ActionType.SWITCH_WINDOW: 0.5,
            ActionType.SHOW_DESKTOP: 1.0,
            ActionType.SCREENSHOT: 1.5,
            ActionType.PLAY_PAUSE: 0.5,
            ActionType.NEXT_TRACK: 0.5,
            ActionType.PREV_TRACK: 0.5,
            ActionType.VOLUME_UP: 0.1,
            ActionType.VOLUME_DOWN: 0.1,
            ActionType.MUTE: 0.5,
        }
        
        self.last_action_time = {}
        self.is_dragging = False
        
        # Mode switch protection
        self.mode_switch_time = 0
        self.mode_switch_cooldown = 1.0  # Seconds after mode switch to ignore actions
        
        # Relative cursor control
        self.last_finger_pos = None
        self.cursor_sensitivity = 2.5  # Multiplier for relative movement
        
    def can_perform_action(self, action: ActionType) -> bool:
        """Check if action cooldown has passed."""
        if action not in self.cooldowns:
            return True
        
        last_time = self.last_action_time.get(action, 0)
        return time.time() - last_time >= self.cooldowns[action]
    
    def is_in_mode_switch_cooldown(self) -> bool:
        """Check if we're still in cooldown after mode switch."""
        return time.time() - self.mode_switch_time < self.mode_switch_cooldown
    
    def record_action(self, action: ActionType):
        """Record action time for cooldown."""
        self.last_action_time[action] = time.time()
    
    def switch_mode(self) -> ControlMode:
        """Switch to next control mode."""
        if not self.can_perform_action(ActionType.MODE_SWITCH):
            return self.current_mode
        
        current_idx = self.MODE_ORDER.index(self.current_mode)
        next_idx = (current_idx + 1) % len(self.MODE_ORDER)
        self.current_mode = self.MODE_ORDER[next_idx]
        
        self.record_action(ActionType.MODE_SWITCH)
        self.mode_switch_time = time.time()
        
        print(f"Mode: {self.current_mode.name}")
        return self.current_mode
    
    def move_cursor(self, x: float, y: float):
        """Move cursor to normalized coordinates with smoothing."""
        # Flip x for mirror effect
        screen_x = (1 - x) * self.screen_width
        screen_y = y * self.screen_height
        
        # Apply smoothing
        smooth_x, smooth_y = self.smoother(screen_x, screen_y)
        
        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
        except Exception:
            pass
    
    def move_cursor_relative(self, x: float, y: float):
        """Move cursor based on relative finger movement."""
        if self.last_finger_pos is None:
            self.last_finger_pos = (x, y)
            return
        
        # Calculate delta (no flip - matches mirrored video)
        dx = (x - self.last_finger_pos[0]) * self.screen_width * self.cursor_sensitivity
        dy = (y - self.last_finger_pos[1]) * self.screen_height * self.cursor_sensitivity
        
        # Update last position
        self.last_finger_pos = (x, y)
        
        # Apply smoothing to delta
        current_x, current_y = pyautogui.position()
        target_x = current_x + dx
        target_y = current_y + dy
        
        smooth_x, smooth_y = self.smoother(target_x, target_y)
        
        # Clamp to screen bounds
        smooth_x = max(0, min(self.screen_width - 1, smooth_x))
        smooth_y = max(0, min(self.screen_height - 1, smooth_y))
        
        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
        except Exception:
            pass
    
    def left_click(self) -> bool:
        """Perform left click."""
        if not self.can_perform_action(ActionType.LEFT_CLICK):
            return False
        
        try:
            pyautogui.click()
            self.record_action(ActionType.LEFT_CLICK)
            return True
        except Exception:
            return False
    
    def right_click(self) -> bool:
        """Perform right click."""
        if not self.can_perform_action(ActionType.RIGHT_CLICK):
            return False
        
        try:
            pyautogui.rightClick()
            self.record_action(ActionType.RIGHT_CLICK)
            return True
        except Exception:
            return False
    
    def double_click(self) -> bool:
        """Perform double click."""
        if not self.can_perform_action(ActionType.DOUBLE_CLICK):
            return False
        
        try:
            pyautogui.doubleClick()
            self.record_action(ActionType.DOUBLE_CLICK)
            return True
        except Exception:
            return False
    
    def middle_click(self) -> bool:
        """Perform middle click."""
        if not self.can_perform_action(ActionType.MIDDLE_CLICK):
            return False
        
        try:
            pyautogui.middleClick()
            self.record_action(ActionType.MIDDLE_CLICK)
            return True
        except Exception:
            return False
    
    def start_drag(self) -> bool:
        """Start dragging."""
        if self.is_dragging:
            return False
        
        try:
            pyautogui.mouseDown()
            self.is_dragging = True
            return True
        except Exception:
            return False
    
    def end_drag(self) -> bool:
        """End dragging."""
        if not self.is_dragging:
            return False
        
        try:
            pyautogui.mouseUp()
            self.is_dragging = False
            return True
        except Exception:
            return False
    
    def scroll(self, angle: float):
        """Scroll based on peace sign angle (legacy)."""
        self.scroll_by_angle(angle)
    
    def scroll_by_angle(self, angle: float):
        """Scroll based on peace sign angle.
        
        Angle range: 0° (horizontal/pointing right) to 90° (vertical/pointing up)
        - Horizontal (0°) = max scroll DOWN
        - Vertical (90°) = max scroll UP
        - 45° = no scroll (neutral)
        
        The angle from tracker is relative to vertical, so we need to convert:
        - tracker angle 0° = vertical (pointing up) = scroll UP max
        - tracker angle 90° = horizontal (pointing right) = scroll DOWN max
        - tracker angle -90° = horizontal (pointing left) = scroll DOWN max
        """
        # Get absolute angle from vertical (0-90 range)
        abs_angle = abs(angle)
        
        # Convert to scroll: 0° vertical = up, 90° horizontal = down
        # Neutral zone around 45°
        if abs_angle < 25:
            # Near vertical = scroll up (speed based on how vertical)
            scroll_speed = int((25 - abs_angle) / 5) + 1  # 1-5
            scroll_speed = min(scroll_speed, 5)
            pyautogui.scroll(scroll_speed)
        elif abs_angle > 65:
            # Near horizontal = scroll down (speed based on how horizontal)
            scroll_speed = int((abs_angle - 65) / 5) + 1  # 1-5
            scroll_speed = min(scroll_speed, 5)
            pyautogui.scroll(-scroll_speed)
    
    def maximize_window(self) -> bool:
        """Maximize current window."""
        if not self.can_perform_action(ActionType.MAXIMIZE):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.hotkey('win', 'up')
            self.record_action(ActionType.MAXIMIZE)
            return True
        except Exception:
            return False
    
    def minimize_window(self) -> bool:
        """Minimize current window."""
        if not self.can_perform_action(ActionType.MINIMIZE):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.hotkey('win', 'down')
            self.record_action(ActionType.MINIMIZE)
            return True
        except Exception:
            return False
    
    def close_window(self) -> bool:
        """Close current window."""
        if not self.can_perform_action(ActionType.CLOSE):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.hotkey('alt', 'F4')
            self.record_action(ActionType.CLOSE)
            return True
        except Exception:
            return False
    
    def switch_window(self) -> bool:
        """Switch to next window."""
        if not self.can_perform_action(ActionType.SWITCH_WINDOW):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.hotkey('alt', 'tab')
            self.record_action(ActionType.SWITCH_WINDOW)
            return True
        except Exception:
            return False
    
    def show_desktop(self) -> bool:
        """Show desktop."""
        if not self.can_perform_action(ActionType.SHOW_DESKTOP):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.hotkey('win', 'd')
            self.record_action(ActionType.SHOW_DESKTOP)
            return True
        except Exception:
            return False
    
    def take_screenshot(self) -> bool:
        """Take screenshot using PrintScreen key."""
        if not self.can_perform_action(ActionType.SCREENSHOT):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.press('printscreen')
            self.record_action(ActionType.SCREENSHOT)
            return True
        except Exception:
            return False
    
    def play_pause(self) -> bool:
        """Play/pause media."""
        if not self.can_perform_action(ActionType.PLAY_PAUSE):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.press('playpause')
            self.record_action(ActionType.PLAY_PAUSE)
            return True
        except Exception:
            return False
    
    def next_track(self) -> bool:
        """Skip to next track."""
        if not self.can_perform_action(ActionType.NEXT_TRACK):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.press('nexttrack')
            self.record_action(ActionType.NEXT_TRACK)
            return True
        except Exception:
            return False
    
    def prev_track(self) -> bool:
        """Skip to previous track."""
        if not self.can_perform_action(ActionType.PREV_TRACK):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.press('prevtrack')
            self.record_action(ActionType.PREV_TRACK)
            return True
        except Exception:
            return False
    
    def volume_up(self) -> bool:
        """Increase volume."""
        if not self.can_perform_action(ActionType.VOLUME_UP):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.press('volumeup')
            self.record_action(ActionType.VOLUME_UP)
            return True
        except Exception:
            return False
    
    def volume_down(self) -> bool:
        """Decrease volume."""
        if not self.can_perform_action(ActionType.VOLUME_DOWN):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.press('volumedown')
            self.record_action(ActionType.VOLUME_DOWN)
            return True
        except Exception:
            return False
    
    def mute(self) -> bool:
        """Toggle mute."""
        if not self.can_perform_action(ActionType.MUTE):
            return False
        if self.is_in_mode_switch_cooldown():
            return False
        
        try:
            pyautogui.press('volumemute')
            self.record_action(ActionType.MUTE)
            return True
        except Exception:
            return False
    
    def reset_smoother(self):
        """Reset cursor smoother and relative tracking."""
        self.smoother.reset()
        self.last_finger_pos = None
