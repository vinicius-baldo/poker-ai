"""
Window Detector: Detects when PokerStars window is activated (hero's turn).
"""
import logging
import platform
import subprocess  # nosec B404
import time

# Windows-specific import (optional)
try:
    import win32gui

    WIN32GUI_AVAILABLE = True
except ImportError:
    WIN32GUI_AVAILABLE = False

logger = logging.getLogger(__name__)


class WindowDetector:
    """Detects when PokerStars window becomes active (hero's turn)."""

    def __init__(self, window_title: str = "PokerStars") -> None:
        """
        Initialize window detector.

        Args:
            window_title: Title of the PokerStars window to monitor
        """
        self.window_title = window_title
        self.last_active_time = 0.0
        self.is_active = False
        self.activation_threshold = 0.1  # 100ms threshold for activation

    def detect_window_activation(self) -> bool:
        """
        Detect if the PokerStars window has just been activated.

        Returns:
            True if window was just activated, False otherwise
        """
        try:
            # Get the active window title
            active_window = self._get_active_window_title()

            # Check if this is a PokerStars game table window (not just the client)
            is_pokerstars_active = self._is_game_table_window(active_window)

            current_time = time.time()

            # Detect activation (wasn't active before, is active now)
            if is_pokerstars_active and not self.is_active:
                # Check if enough time has passed since last activation
                if current_time - self.last_active_time > self.activation_threshold:
                    self.last_active_time = current_time
                    self.is_active = True
                    logger.info(f"🎯 PokerStars GAME TABLE activated: {active_window}")
                    return True

            # Update state
            self.is_active = is_pokerstars_active

            return False

        except Exception as e:
            logger.error(f"Error detecting window activation: {e}")
            return False

    def _is_game_table_window(self, window_title: str) -> bool:
        """
        Check if the window title indicates a PokerStars game table.

        Args:
            window_title: The window title to check

        Returns:
            True if it's a game table window, False otherwise
        """
        if not window_title:
            return False

        title_lower = window_title.lower()

        # Language-agnostic PokerStars game table detection
        # Look for PokerStars-specific patterns that work across all languages

        # 1. Must contain currency symbols (buy-ins, blinds, etc.)
        has_currency = any(
            symbol in title_lower for symbol in ["$", "€", "£", "₽", "¥"]
        )

        # 2. Must contain PokerStars-specific separators and patterns
        has_pokerstars_patterns = any(
            pattern in title_lower
            for pattern in [
                " - ",  # PokerStars uses " - " to separate table info
                "ante",  # Ante games (common across languages)
                "gtd",  # Guaranteed tournaments (abbreviation)
                "blinds",  # Blind levels (common term)
            ]
        )

        # 3. Must contain game structure indicators
        has_game_indicators = any(
            pattern in title_lower
            for pattern in [
                "pokerstars",  # The client name
                "mesa",  # Table (common in many languages)
                "table",  # Table (common in many languages)
            ]
        )

        # Return True if we have currency + either PokerStars patterns or game \
        # indicators
        return has_currency and (has_pokerstars_patterns or has_game_indicators)

    def _get_active_window_title(self) -> str:
        """
        Get the title of the currently active window.

        Returns:
            Window title as string
        """
        try:
            if platform.system() == "Darwin":  # macOS
                return self._get_active_window_macos()
            elif platform.system() == "Windows":
                return self._get_active_window_windows()
            elif platform.system() == "Linux":
                return self._get_active_window_linux()
            else:
                logger.warning(f"Unsupported platform: {platform.system()}")
                return ""

        except Exception as e:
            logger.error(f"Error getting active window title: {e}")
            return ""

    def _get_active_window_macos(self) -> str:
        """Get active window title on macOS."""
        try:
            # Use AppleScript to get active window title
            script = """
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is \
                    true
                try
                    set windowList to windows of process frontApp
                    if (count of windowList) > 0 then
                        set windowTitle to name of item 1 of windowList
                        return windowTitle
                    else
                        return frontApp
                    end if
                on error
                    return frontApp
                end try
            end tell
            """

            result = subprocess.run(  # nosec
                ["osascript", "-e", script], capture_output=True, text=True, timeout=1
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(f"AppleScript failed: {result.stderr}")
                return ""

        except Exception as e:
            logger.error(f"Error getting active window on macOS: {e}")
            return ""

    def _get_active_window_windows(self) -> str:
        """Get active window title on Windows."""
        if not WIN32GUI_AVAILABLE:
            logger.warning("win32gui not available on Windows")
            return ""

        try:

            def enum_windows_callback(hwnd: int, windows: list) -> bool:
                if win32gui.IsWindowVisible(hwnd):
                    window_title = str(win32gui.GetWindowText(hwnd))
                    if window_title:
                        windows.append((hwnd, window_title))
                return True

            windows: list = []
            win32gui.EnumWindows(enum_windows_callback, windows)

            # Find the active window
            active_hwnd = win32gui.GetForegroundWindow()
            for hwnd, title in windows:
                if hwnd == active_hwnd:
                    return str(title)

            return ""

        except Exception as e:
            logger.error(f"Error getting active window on Windows: {e}")
            return ""

    def _get_active_window_linux(self) -> str:
        """Get active window title on Linux."""
        try:
            # Try xdotool first
            try:
                result = subprocess.run(  # nosec
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )

                if result.returncode == 0:
                    return result.stdout.strip()
            except FileNotFoundError:
                pass

            # Fallback to wmctrl
            try:
                result = subprocess.run(  # nosec
                    ["wmctrl", "-a", "."], capture_output=True, text=True, timeout=1
                )

                if result.returncode == 0:
                    # Parse wmctrl output to get active window
                    result = subprocess.run(  # nosec
                        ["wmctrl", "-l"], capture_output=True, text=True, timeout=1
                    )

                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        for line in lines:
                            if line.startswith("0x"):
                                parts = line.split()
                                if len(parts) >= 4:
                                    return " ".join(parts[3:])
            except FileNotFoundError:
                pass

            logger.warning("Neither xdotool nor wmctrl available on Linux")
            return ""

        except Exception as e:
            logger.error(f"Error getting active window on Linux: {e}")
            return ""

    def set_activation_threshold(self, threshold: float) -> None:
        """
        Set the activation threshold in seconds.

        Args:
            threshold: Minimum time between activations in seconds
        """
        self.activation_threshold = threshold

    def get_status(self) -> dict:
        """
        Get current status of the window detector.

        Returns:
            Dictionary with current status
        """
        return {
            "window_title": self.window_title,
            "is_active": self.is_active,
            "last_active_time": self.last_active_time,
            "activation_threshold": self.activation_threshold,
        }

    def reset(self) -> None:
        """Reset the detector state."""
        self.last_active_time = 0.0
        self.is_active = False


def test_window_detector() -> None:
    """Test the window detector functionality."""
    print("🎯 Testing Window Detector...")
    print("=" * 40)

    detector = WindowDetector("PokerStars")

    print("Monitoring for PokerStars window activation...")
    print("Please activate the PokerStars window to test detection.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            if detector.detect_window_activation():
                print("✅ PokerStars window activated!")
                print(f"Status: {detector.get_status()}")

            time.sleep(0.1)  # Check every 100ms

    except KeyboardInterrupt:
        print("\nTest stopped by user")


if __name__ == "__main__":
    test_window_detector()
