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
    """Combines One Euro Filters for X and Y."""
    
    def __init__(self):
        self.euro_x = OneEuroFilter(mincutoff=1.0, beta=0.5)
        self.euro_y = OneEuroFilter(mincutoff=1.0, beta=0.5)
        self.history = deque(maxlen=10)
    
    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        current_time = time.time()
        x_smooth = self.euro_x(x, current_time)
        y_smooth = self.euro_y(y, current_time)
        self.history.append((x_smooth, y_smooth))
        return x_smooth, y_smooth
    
    def reset(self):
        self.euro_x.reset()
        self.euro_y.reset()
        self.history.clear()
