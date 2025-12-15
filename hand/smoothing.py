"""
Smoothing filters for cursor movement.

This module implements adaptive cursor smoothing to reduce jitter while
maintaining responsiveness. Key components:

1. One Euro Filter: Adaptive low-pass filter that automatically adjusts
   based on movement speed:
   - Slow movements → strong smoothing (reduces jitter)
   - Fast movements → minimal smoothing (stays responsive)

2. AdaptiveSmoother: Combines One Euro Filters for X/Y with:
   - Deadzone filtering (ignores sub-pixel jitter)
   - Edge-aware smoothing (extra smoothing near screen edges)
   - Position history for continuity

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import numpy as np
import time
from collections import deque
from typing import Optional, Tuple


class OneEuroFilter:
    """
    One Euro Filter for smooth, low-latency signal processing.
    
    The filter adapts its cutoff frequency based on signal velocity:
    - High velocity → high cutoff → less smoothing → low latency
    - Low velocity → low cutoff → more smoothing → less jitter
    
    Parameters:
        freq: Sampling frequency in Hz (updated dynamically)
        mincutoff: Minimum cutoff frequency (lower = smoother)
        beta: Speed coefficient (higher = more responsive to fast moves)
        dcutoff: Cutoff frequency for derivative filtering
    """
    
    def __init__(self, freq: float = 60.0, mincutoff: float = 1.0, 
                 beta: float = 0.007, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def _alpha(self, cutoff: float) -> float:
        """Calculate smoothing factor alpha from cutoff frequency."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter input value x at time t.
        
        Args:
            x: Input value to filter
            t: Optional timestamp (uses internal timing if None)
            
        Returns:
            Filtered value
        """
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        # Update frequency estimate from time delta
        if t is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if dt > 0:
                self.freq = 1.0 / dt
        self.t_prev = t
        
        # Filter the derivative (velocity)
        a_d = self._alpha(self.dcutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat
        
        # Adaptive cutoff based on velocity
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        
        # Filter the value
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat
    
    def reset(self):
        """Reset filter state (call when tracking is lost)."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class AdaptiveSmoother:
    """
    Advanced cursor smoother with position-aware filtering.
    
    Combines multiple smoothing techniques:
    1. One Euro Filter for each axis (adaptive low-pass)
    2. Deadzone filtering (ignores sub-threshold movements)
    3. Edge-aware smoothing (extra smoothing near screen edges)
    
    The edge smoothing helps with:
    - Camera tracking instability at frame edges
    - Non-linear coordinate mapping at extremes
    - Natural hand position limitations
    
    Attributes:
        screen_width, screen_height: Screen dimensions for edge detection
        deadzone: Minimum movement in pixels to register (default: 2.0)
        edge_margin: Screen edge zone as fraction (default: 0.15 = 15%)
        edge_smoothing_factor: Extra smoothing at edges (default: 0.6)
    """
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        """
        Initialize the adaptive smoother.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        # One Euro Filters with tuned parameters
        # Lower mincutoff = more smoothing, higher beta = more responsive
        self.euro_x = OneEuroFilter(mincutoff=0.8, beta=0.4, dcutoff=1.0)
        self.euro_y = OneEuroFilter(mincutoff=0.8, beta=0.4, dcutoff=1.0)
        self.history = deque(maxlen=10)
        
        # Screen dimensions for edge detection
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Deadzone: ignore movements smaller than this (reduces jitter)
        self.deadzone = 2.0  # pixels
        
        # Previous smoothed position for continuity
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        
        # Edge smoothing parameters
        self.edge_margin = 0.15  # 15% of screen considered "edge zone"
        self.edge_smoothing_factor = 0.6  # Extra smoothing strength (0-1, lower = smoother)
    
    def _is_near_edge(self, x: float, y: float) -> Tuple[bool, float]:
        """
        Check if position is near screen edge and calculate smoothing factor.
        
        Args:
            x, y: Screen coordinates in pixels
            
        Returns:
            Tuple of (is_near_edge, smoothing_factor)
            smoothing_factor ranges from edge_smoothing_factor at edge to 1.0 at margin boundary
        """
        # Calculate normalized position (0-1)
        norm_x = x / self.screen_width
        norm_y = y / self.screen_height
        
        # Distance from nearest edge (0 = at edge, 0.5 = center)
        edge_dist_x = min(norm_x, 1 - norm_x)
        edge_dist_y = min(norm_y, 1 - norm_y)
        min_edge_dist = min(edge_dist_x, edge_dist_y)
        
        # Apply extra smoothing within edge margin
        if min_edge_dist < self.edge_margin:
            # Smoothing increases as we get closer to edge
            edge_factor = min_edge_dist / self.edge_margin  # 0 at edge, 1 at margin
            smoothing_factor = self.edge_smoothing_factor + (1 - self.edge_smoothing_factor) * edge_factor
            return True, smoothing_factor
        
        return False, 1.0
    
    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply smoothing to cursor position.
        
        Args:
            x, y: Target cursor position in pixels
            
        Returns:
            Smoothed (x, y) position
        """
        current_time = time.time()
        
        # Step 1: Apply One Euro filter
        x_filtered = self.euro_x(x, current_time)
        y_filtered = self.euro_y(y, current_time)
        
        # Step 2: Apply deadzone filtering
        if self.prev_x is not None and self.prev_y is not None:
            dx = abs(x_filtered - self.prev_x)
            dy = abs(y_filtered - self.prev_y)
            
            # Dampen sub-deadzone movements by 70%
            if dx < self.deadzone:
                x_filtered = self.prev_x + (x_filtered - self.prev_x) * 0.3
            if dy < self.deadzone:
                y_filtered = self.prev_y + (y_filtered - self.prev_y) * 0.3
        
        # Step 3: Apply edge smoothing
        near_edge, smoothing_factor = self._is_near_edge(x_filtered, y_filtered)
        
        if near_edge and self.prev_x is not None and self.prev_y is not None:
            # Extra exponential smoothing at edges
            x_filtered = self.prev_x + (x_filtered - self.prev_x) * smoothing_factor
            y_filtered = self.prev_y + (y_filtered - self.prev_y) * smoothing_factor
        
        # Update state
        self.prev_x = x_filtered
        self.prev_y = y_filtered
        self.history.append((x_filtered, y_filtered))
        
        return x_filtered, y_filtered
    
    def reset(self):
        """Reset smoother state (call when hand tracking is lost)."""
        self.euro_x.reset()
        self.euro_y.reset()
        self.history.clear()
        self.prev_x = None
        self.prev_y = None
