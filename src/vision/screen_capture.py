"""
Screen Capture: Real-time capture of poker table screenshots.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Real-time screen capture for poker table monitoring."""

    def __init__(self, monitor_id: int = 1) -> None:
        """Initialize screen capture."""
        self.monitor_id = monitor_id
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_id]
        self.last_capture_time: float = 0.0
        self.capture_interval: float = 0.5  # Capture every 500ms

    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture current screen as numpy array."""
        try:
            screenshot = self.sct.grab(self.monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return np.array(img)
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None

    def capture_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Capture specific region of the screen.

        Args:
            region: (x, y, width, height) coordinates

        Returns:
            Numpy array of the captured region
        """
        try:
            x, y, width, height = region
            monitor_region = {"top": y, "left": x, "width": width, "height": height}

            screenshot = self.sct.grab(monitor_region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return np.array(img)
        except Exception as e:
            logger.error(f"Error capturing region {region}: {e}")
            return None

    def capture_with_rate_limit(self) -> Optional[np.ndarray]:
        """Capture screen with rate limiting to avoid excessive captures."""
        current_time = time.time()

        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            return self.capture_screen()

        return None

    def set_capture_interval(self, interval: float) -> None:
        """Set the capture interval in seconds."""
        self.capture_interval = interval

    def get_monitor_info(self) -> dict:
        """Get information about the current monitor."""
        return {
            "id": self.monitor_id,
            "width": self.monitor["width"],
            "height": self.monitor["height"],
            "top": self.monitor["top"],
            "left": self.monitor["left"],
        }

    def list_monitors(self) -> List[Dict[str, Any]]:
        """List all available monitors."""
        monitors = self.sct.monitors
        if isinstance(monitors, list):
            return monitors
        return []

    def close(self) -> None:
        """Close the screen capture instance."""
        self.sct.close()


class PokerTableCapture:
    """Specialized capture for poker table regions."""

    def __init__(
        self, table_region: Optional[Tuple[int, int, int, int]] = None
    ) -> None:
        """
        Initialize poker table capture.

        Args:
            table_region: (x, y, width, height) of poker table area
        """
        self.screen_capture = ScreenCapture()
        self.table_region = table_region
        self.is_capturing = False
        self.capture_thread: Optional[Any] = None

    def set_table_region(self, region: Tuple[int, int, int, int]) -> None:
        """Set the poker table region to capture."""
        self.table_region = region
        logger.info(f"Table region set to: {region}")

    def capture_table(self) -> Optional[np.ndarray]:
        """Capture the poker table region."""
        if self.table_region:
            return self.screen_capture.capture_region(self.table_region)
        else:
            return self.screen_capture.capture_screen()

    def start_continuous_capture(self, callback: Any, interval: float = 1.0) -> None:
        """
        Start continuous capture with callback.

        Args:
            callback: Function to call with captured image
            interval: Capture interval in seconds
        """
        import threading

        self.screen_capture.set_capture_interval(interval)
        self.is_capturing = True

        def capture_loop() -> None:
            while self.is_capturing:
                img = self.screen_capture.capture_with_rate_limit()
                if img is not None:
                    try:
                        callback(img)
                    except Exception as e:
                        logger.error(f"Error in capture callback: {e}")
                time.sleep(0.1)  # Small sleep to prevent excessive CPU usage

        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Started continuous capture")

    def stop_continuous_capture(self) -> None:
        """Stop continuous capture."""
        self.is_capturing = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
        logger.info("Stopped continuous capture")

    def calibrate_table_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Interactive calibration to select poker table region.

        Returns:
            Selected region coordinates (x, y, width, height)
        """

        def mouse_callback(
            event: int, x: int, y: int, flags: int, param: Dict[str, Any]
        ) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                param["start_point"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                param["end_point"] = (x, y)
                param["selection_done"] = True

        # Capture full screen for calibration
        full_screen = self.screen_capture.capture_screen()
        if full_screen is None:
            logger.error("Failed to capture screen for calibration")
            return None

        # Convert BGR to RGB for display
        full_screen_rgb = cv2.cvtColor(full_screen, cv2.COLOR_BGR2RGB)

        # Create window and set mouse callback
        cv2.namedWindow("Select Poker Table Region", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            "Select Poker Table Region",
            mouse_callback,
            {"start_point": None, "end_point": None, "selection_done": False},
        )

        selection_data: Dict[str, Any] = {
            "start_point": None,
            "end_point": None,
            "selection_done": False,
        }

        while not selection_data["selection_done"]:
            display_img = full_screen_rgb.copy()

            # Draw selection rectangle if started
            if selection_data["start_point"]:
                cv2.rectangle(
                    display_img,
                    selection_data["start_point"],
                    (
                        selection_data["start_point"][0] + 50,
                        selection_data["start_point"][1] + 50,
                    ),
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Select Poker Table Region", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                break

        cv2.destroyAllWindows()

        if selection_data["start_point"] and selection_data["end_point"]:
            start_point = selection_data["start_point"]
            end_point = selection_data["end_point"]
            if start_point and end_point:
                x1, y1 = start_point
                x2, y2 = end_point

            # Ensure coordinates are in correct order
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            region = (x, y, width, height)
            self.set_table_region(region)
            return region

        return None

    def close(self) -> None:
        """Close the poker table capture."""
        self.stop_continuous_capture()
        self.screen_capture.close()


def test_screen_capture() -> None:
    """Test the screen capture functionality."""
    print("🖥️ Testing Screen Capture...")
    print("=" * 40)

    # Test basic capture
    capture = ScreenCapture()

    # List monitors
    monitors = capture.list_monitors()
    print(f"📺 Found {len(monitors)} monitor(s):")
    for i, monitor in enumerate(monitors):
        print(f"  Monitor {i}: {monitor['width']}x{monitor['height']}")

    # Get current monitor info
    info = capture.get_monitor_info()
    print(f"🎯 Current monitor: {info['width']}x{info['height']}")

    # Test capture
    print("📸 Capturing screen...")
    img = capture.capture_screen()
    if img is not None:
        print(f"✅ Captured image: {img.shape}")

        # Save test image
        cv2.imwrite("test_capture.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("💾 Saved test image as 'test_capture.png'")
    else:
        print("❌ Failed to capture screen")

    capture.close()
    print("✅ Screen capture test completed!")


if __name__ == "__main__":
    test_screen_capture()
