# Wision 🖐️

> **Control your computer with hand gestures using your webcam**

Wision is a real-time hand gesture recognition system that lets you control your Windows PC without touching the mouse or keyboard. It uses computer vision and machine learning to track your hand and translate gestures into system actions like cursor movement, clicks, scrolling, window management, and media control.

---

## ✨ Features

- **🎯 Real-time hand tracking** — Uses Google MediaPipe for accurate 21-point hand landmark detection
- **🤌 12 distinct gestures** — Comprehensive gesture vocabulary for varied control options
- **🔄 4 control modes** — Cursor, Scroll, Window, and Media modes for different use cases
- **🖱️ Smooth cursor control** — One Euro Filter with position-aware edge smoothing and deadzone filtering
- **⏱️ Time-based gesture validation** — Prevents false triggers with configurable duration thresholds
- **🎛️ Action cooldowns** — Prevents accidental repeated actions (e.g., double clicks)
- **⚡ Low latency** — Optimized pipeline for responsive interaction
- **🎨 Visual feedback** — On-screen UI shows current mode and detected gesture
- **🛠️ Testing utility** — Built-in calibration tool for debugging gesture detection

---

## 🎮 Control Modes

Switch between modes using the **OK Sign** gesture (hold for ~0.3 seconds):

| Mode | Purpose | Key Gestures |
|------|---------|--------------|
| **CURSOR** | Mouse control | Point to move, L-sign/pinch to click, peace sign to scroll |
| **SCROLL** | Dedicated scrolling | Peace sign angle controls scroll direction/speed |
| **WINDOW** | Window management | Open palm = maximize, grab = minimize, rock = Alt+Tab |
| **MEDIA** | Media playback | Open palm = play/pause, pinch = volume, rock = volume down |

---

## 🤚 Gesture Reference

| Gesture | Hand Position | Description |
|---------|---------------|-------------|
| **OK Sign** | Thumb + index form a circle, other fingers extended | Mode switcher |
| **Open Palm** | All 5 fingers fully extended | Context-dependent action |
| **Grab/Fist** | 3+ fingers curled into palm | Drag, minimize, or mute |
| **Index Up** | Only index finger extended, thumb relaxed | Cursor movement |
| **L Sign** | Thumb + index fully extended (L shape) | Cursor movement + click |
| **Peace Sign** | Index + middle extended (V shape) | Scroll or navigation |
| **Rock Sign** | Index + pinky extended, thumb tucked | Right-click or volume |
| **Call Me** | Thumb + pinky extended, others curled | Double-click or close |
| **Pinch Middle** | Thumb touches middle finger | Left-click or volume up |

---

## 📋 Gesture-to-Action Mapping

<details>
<summary><b>CURSOR Mode</b></summary>

| Gesture | Action | Threshold |
|---------|--------|-----------|
| Index Up | Move cursor | Immediate |
| L Sign | Move cursor + Left click | 0.2s + 3 frames |
| Pinch Middle | Left click | 0.15s + 3 frames |
| Rock Sign | Right click | 0.2s + 3 frames |
| Call Me | Double click | 0.25s + 5 frames |
| Peace Sign | Scroll (angle-based) | 0.1s hold |
| Ring Curl | Middle click | 0.2s + 3 frames |

</details>

<details>
<summary><b>SCROLL Mode</b></summary>

| Gesture | Action | Threshold |
|---------|--------|-----------|
| Index Up | Move cursor | Immediate |
| Peace Sign | Scroll (vertical = up, horizontal = down) | 0.1s hold |

</details>

<details>
<summary><b>WINDOW Mode</b></summary>

| Gesture | Action | Threshold |
|---------|--------|-----------|
| Open Palm | Maximize window (Win+Up) | 0.3s + 5 frames |
| Grab | Minimize window (Win+Down) | 0.3s + 5 frames |
| Rock Sign | Switch window (Alt+Tab) | 0.2s + 3 frames |
| Pinch Middle | Show desktop (Win+D) | 0.2s + 3 frames |
| Call Me | Close window (Alt+F4) | 0.5s + 8 frames |
| Peace Sign | Take screenshot | 0.3s + 5 frames |

</details>

<details>
<summary><b>MEDIA Mode</b></summary>

| Gesture | Action | Threshold |
|---------|--------|-----------|
| Open Palm | Play/Pause | 0.2s + 3 frames |
| Peace Sign | Previous track | 0.2s + 3 frames |
| Pinch Middle | Volume up | 0.1s hold |
| Rock Sign | Volume down | 0.1s hold |
| Grab | Mute/Unmute | 0.3s + 5 frames |

</details>

---

## 🚀 Quick Start

### Prerequisites

- **Windows 10/11** (PyAutoGUI uses Windows APIs)
- **Python 3.9+** (tested with 3.11)
- **Webcam** (built-in or external)

### Installation

```bash
# Clone the repository
git clone https://github.com/code-pasu/Wision.git
cd Wision

# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Controller

```bash
# Run the main gesture controller
python run_hand.py

# Or run as a module
python -m hand
```

### Testing Gestures (Safe Mode)

The testing utility shows what gestures are detected **without** triggering any system actions. Great for calibration and debugging:

```bash
python test_gestures.py
```

### Controls

- **OK Sign (hold 0.3s)** — Switch between control modes
- **Q or ESC** — Quit the application

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Webcam    │───▶│  Tracker    │───▶│ Recognizer  │───▶│   Actions   │
│  (OpenCV)   │    │ (MediaPipe) │    │  (Rules)    │    │ (PyAutoGUI) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │                  │
                          ▼                  ▼                  ▼
                    21 landmarks      Gesture label      OS commands
                    per frame         + stability        (mouse/keys)
                                      + duration
```

### Project Structure

```
Wision/
├── hand/                    # Main package
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # Module entry point
│   ├── hand.py              # Documentation + entry point
│   ├── gestures.py          # Gesture/Mode enums and state tracking
│   ├── tracker.py           # Hand tracking with MediaPipe
│   ├── recognizer.py        # Gesture classification with robust detection
│   ├── actions.py           # System control via PyAutoGUI with cooldowns
│   ├── smoothing.py         # One Euro Filter + edge-aware smoothing
│   └── controller.py        # Main application loop with time-based validation
├── run_hand.py              # Quick launcher
├── test_gestures.py         # Gesture testing/calibration utility
├── requirements.txt         # Python dependencies
├── assets/                  # Images and media
│   └── gestures/            # Gesture reference images
└── README.md                # This file
```

---

## 🔧 Configuration

### Detection Thresholds

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `detection_confidence` | `tracker.py` | 0.7 | Min confidence for hand detection |
| `tracking_confidence` | `tracker.py` | 0.7 | Min confidence for landmark tracking |
| `OK_SIGN_THRESHOLD` | `recognizer.py` | 0.05 | Max thumb-index distance for OK sign |
| `PINCH_THRESHOLD` | `recognizer.py` | 0.06 | Max distance for pinch detection |
| `MIN_GESTURE_DURATION` | `recognizer.py` | 0.15s | Default time threshold for gestures |
| `STABILITY_FRAMES` | `recognizer.py` | 3 | Min consecutive frames for stability |

### Action Cooldowns

| Action | Cooldown | Description |
|--------|----------|-------------|
| Left Click | 0.5s | Prevents accidental double-clicks |
| Right Click | 0.7s | Longer to avoid misclicks |
| Mode Switch | 0.8s | Prevents rapid mode cycling |
| Close Window | 1.0s | Safety delay for destructive action |

### Cursor Smoothing

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `cursor_sensitivity` | `actions.py` | 2.5 | Movement multiplier |
| `mincutoff` | `smoothing.py` | 0.8 | Base smoothing (lower = smoother) |
| `beta` | `smoothing.py` | 0.4 | Speed adaptation factor |
| `deadzone` | `smoothing.py` | 2.0px | Ignore movements below this |
| `edge_margin` | `smoothing.py` | 15% | Screen edge zone for extra smoothing |

---

## 🔬 Gesture Detection Details

### Robust Detection Features

The gesture recognizer uses multiple validation layers:

1. **Basic Finger State Check** — Extension/curl state of each finger
2. **Geometric Validation** — Joint angles, distances, and positions
3. **Frame Stability** — Gesture must persist for N consecutive frames
4. **Time Duration** — Gesture must be held for minimum seconds
5. **Priority Ordering** — Most specific gestures checked first

### L Sign Detection
- Thumb angle at IP joint > 155° (fully straight)
- Thumb tip distance from index MCP > 0.12 (clearly extended outward)
- Thumb tip distance from wrist > 0.15 (not curled back)
- Index extended, middle/ring/pinky curled

### Rock Sign Detection
- Thumb NOT extended and tucked near palm (< 0.12 from index MCP)
- Index and pinky clearly extended
- Middle and ring curled
- Index-pinky spread > 0.08 (fingers apart)

### Index Up Detection
- Index finger extended with straight joints
- Index tip above index MCP (pointing up)
- Middle, ring, pinky all curled
- Thumb state ignored (allows natural hand position)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [MediaPipe](https://mediapipe.dev/) | ≥0.10.0 | Hand landmark detection (ML model) |
| [OpenCV](https://opencv.org/) | ≥4.8.0 | Video capture and image processing |
| [NumPy](https://numpy.org/) | ≥1.24.0 | Numerical computations |
| [PyAutoGUI](https://pyautogui.readthedocs.io/) | ≥0.9.54 | Mouse/keyboard control |

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add multi-hand support for two-handed gestures
- [ ] Create a GUI for configuration
- [ ] Add gesture recording and custom gesture training
- [ ] Port to macOS/Linux
- [ ] Add voice command integration
- [ ] Implement machine learning-based gesture classification

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Google MediaPipe](https://mediapipe.dev/) for the hand tracking model
- [One Euro Filter](https://cristal.univ-lille.fr/~casiez/1euro/) algorithm for smooth cursor movement
- The open-source community for inspiration and tools

---

<p align="center">
  Made with ❤️ and 🤚
</p>
