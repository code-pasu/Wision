"""
Smoothing filters for cursor movement.

Implements the One Euro Filter algorithm for adaptive low-pass filtering.
This filter provides:
- Strong smoothing for slow movements (reduces jitter)
- Minimal lag for fast movements (stays responsive)

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import numpy as np
import time
from collections import deque
from typing import Optional, Tuple


class OneEuroFilter:
    """One Euro Filter for smooth cursor control."""
    
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
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        if t is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if dt > 0:
                self.freq = 1.0 / dt
        self.t_prev = t
        
        a_d = self._alpha(self.dcutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat
        
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class AdaptiveSmoother:
    """Combines One Euro Filters for X and Y with position-aware smoothing."""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        # Adjusted parameters for smoother movement
        # Lower mincutoff = more smoothing, higher beta = more responsive to fast moves
        self.euro_x = OneEuroFilter(mincutoff=0.8, beta=0.4, dcutoff=1.0)
        self.euro_y = OneEuroFilter(mincutoff=0.8, beta=0.4, dcutoff=1.0)
        self.history = deque(maxlen=10)
        
        # Screen dimensions for edge detection
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Deadzone threshold - ignore very small movements (jitter)
        self.deadzone = 2.0  # pixels
        
        # Previous smoothed position
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        
        # Extra smoothing for edge regions
        self.edge_margin = 0.15  # 15% of screen height/width considered "edge"
        self.edge_smoothing_factor = 0.6  # Additional smoothing at edges (0-1, lower = smoother)
    
    def _is_near_edge(self, x: float, y: float) -> Tuple[bool, float]:
        """Check if position is near screen edge and return smoothing factor."""
        # Calculate normalized position (0-1)
        norm_x = x / self.screen_width
        norm_y = y / self.screen_height
        
        # Check distance from edges
        edge_dist_x = min(norm_x, 1 - norm_x)  # Distance from left/right edge
        edge_dist_y = min(norm_y, 1 - norm_y)  # Distance from top/bottom edge
        min_edge_dist = min(edge_dist_x, edge_dist_y)
        
        # If within edge margin, apply extra smoothing
        if min_edge_dist < self.edge_margin:
            # Smoothing increases as we get closer to edge
            edge_factor = min_edge_dist / self.edge_margin  # 0 at edge, 1 at margin boundary
            smoothing_factor = self.edge_smoothing_factor + (1 - self.edge_smoothing_factor) * edge_factor
            return True, smoothing_factor
        
        return False, 1.0
    
    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        current_time = time.time()
        
        # Apply One Euro filter first
        x_filtered = self.euro_x(x, current_time)
        y_filtered = self.euro_y(y, current_time)
        
        # Apply deadzone filtering
        if self.prev_x is not None and self.prev_y is not None:
            dx = abs(x_filtered - self.prev_x)
            dy = abs(y_filtered - self.prev_y)
            
            # If movement is below deadzone, reduce it significantly
            if dx < self.deadzone:
                x_filtered = self.prev_x + (x_filtered - self.prev_x) * 0.3
            if dy < self.deadzone:
                y_filtered = self.prev_y + (y_filtered - self.prev_y) * 0.3
        
        # Check if near edge and apply extra smoothing
        near_edge, smoothing_factor = self._is_near_edge(x_filtered, y_filtered)
        
        if near_edge and self.prev_x is not None and self.prev_y is not None:
            # Apply additional exponential smoothing at edges
            x_filtered = self.prev_x + (x_filtered - self.prev_x) * smoothing_factor
            y_filtered = self.prev_y + (y_filtered - self.prev_y) * smoothing_factor
        
        # Update previous position
        self.prev_x = x_filtered
        self.prev_y = y_filtered
        
        self.history.append((x_filtered, y_filtered))
        return x_filtered, y_filtered
    
    def reset(self):
        self.euro_x.reset()
        self.euro_y.reset()
        self.history.clear()
        self.prev_x = None
        self.prev_y = None
