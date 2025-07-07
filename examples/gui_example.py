#!/usr/bin/env python3
"""
Simple GUI example for Poker AI Assistant using tkinter.
"""
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Local imports after path modification
from main_poker_assistant import PokerAssistant  # noqa: E402


class PokerAIGUI:
    """Simple GUI for Poker AI Assistant."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the GUI."""
        self.root = root
        self.root.title("Poker AI Assistant")
        self.root.geometry("800x600")

        # Initialize assistant
        self.assistant = PokerAssistant()

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self) -> None:
        """Create and arrange GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="wens")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Image selection
        ttk.Label(main_frame, text="Poker Table Image:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.image_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.image_path_var, width=50).grid(
            row=0, column=1, sticky="we", padx=5
        )
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(
            row=0, column=2, padx=5
        )

        # Analyze button
        ttk.Button(
            main_frame, text="Analyze Situation", command=self.analyze_situation
        ).grid(row=1, column=0, columnspan=3, pady=10)

        # Create notebook for different sections
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=2, column=0, columnspan=3, sticky="wens", pady=10)
        main_frame.rowconfigure(2, weight=1)

        # AI Analysis tab
        ai_frame = ttk.Frame(notebook)
        notebook.add(ai_frame, text="AI Analysis")
        self.create_ai_tab(ai_frame)

        # Table Info tab
        table_frame = ttk.Frame(notebook)
        notebook.add(table_frame, text="Table Info")
        self.create_table_tab(table_frame)

        # Cards tab
        cards_frame = ttk.Frame(notebook)
        notebook.add(cards_frame, text="Cards")
        self.create_cards_tab(cards_frame)

        # Raw Data tab
        raw_frame = ttk.Frame(notebook)
        notebook.add(raw_frame, text="Raw Data")
        self.create_raw_tab(raw_frame)

    def create_ai_tab(self, parent: ttk.Frame) -> None:
        """Create AI analysis tab."""
        # Recommendation section
        rec_frame = ttk.LabelFrame(parent, text="Recommendation", padding="10")
        rec_frame.pack(fill=tk.X, padx=5, pady=5)

        self.recommendation_var = tk.StringVar(value="No analysis yet")
        self.confidence_var = tk.StringVar(value="0%")
        self.reasoning_var = tk.StringVar(value="")

        ttk.Label(rec_frame, text="Action:").grid(row=0, column=0, sticky="w")
        self.recommendation_label = ttk.Label(
            rec_frame, textvariable=self.recommendation_var, font=("Arial", 14, "bold")
        )
        self.recommendation_label.grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(rec_frame, text="Confidence:").grid(row=1, column=0, sticky="w")
        self.confidence_label = ttk.Label(
            rec_frame, textvariable=self.confidence_var, font=("Arial", 12)
        )
        self.confidence_label.grid(row=1, column=1, sticky="w", padx=10)

        ttk.Label(rec_frame, text="Reasoning:").grid(row=2, column=0, sticky="w")
        reasoning_text = tk.Text(rec_frame, height=4, width=50, wrap=tk.WORD)
        reasoning_text.grid(row=2, column=1, sticky="we", padx=10, pady=5)
        self.reasoning_text = reasoning_text

        # Metrics section
        metrics_frame = ttk.LabelFrame(parent, text="Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        self.expected_value_var = tk.StringVar(value="$0.00")
        self.risk_level_var = tk.StringVar(value="Unknown")
        self.hand_strength_var = tk.StringVar(value="Unknown")
        self.pot_odds_var = tk.StringVar(value="0:1")

        ttk.Label(metrics_frame, text="Expected Value:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(metrics_frame, textvariable=self.expected_value_var).grid(
            row=0, column=1, sticky="w", padx=10
        )

        ttk.Label(metrics_frame, text="Risk Level:").grid(row=1, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.risk_level_var).grid(
            row=1, column=1, sticky="w", padx=10
        )

        ttk.Label(metrics_frame, text="Hand Strength:").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(metrics_frame, textvariable=self.hand_strength_var).grid(
            row=2, column=1, sticky="w", padx=10
        )

        ttk.Label(metrics_frame, text="Pot Odds:").grid(row=3, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.pot_odds_var).grid(
            row=3, column=1, sticky="w", padx=10
        )

    def create_table_tab(self, parent: ttk.Frame) -> None:
        """Create table info tab."""
        table_frame = ttk.LabelFrame(parent, text="Table Information", padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pot_size_var = tk.StringVar(value="$0.00")
        self.current_bet_var = tk.StringVar(value="$0.00")
        self.street_var = tk.StringVar(value="Unknown")

        ttk.Label(table_frame, text="Pot Size:").grid(row=0, column=0, sticky="w")
        ttk.Label(
            table_frame, textvariable=self.pot_size_var, font=("Arial", 12, "bold")
        ).grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(table_frame, text="Current Bet:").grid(row=1, column=0, sticky="w")
        ttk.Label(table_frame, textvariable=self.current_bet_var).grid(
            row=1, column=1, sticky="w", padx=10
        )

        ttk.Label(table_frame, text="Street:").grid(row=2, column=0, sticky="w")
        ttk.Label(table_frame, textvariable=self.street_var).grid(
            row=2, column=1, sticky="w", padx=10
        )

    def create_cards_tab(self, parent: ttk.Frame) -> None:
        """Create cards tab."""
        cards_frame = ttk.LabelFrame(parent, text="Detected Cards", padding="10")
        cards_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Hole cards
        hole_frame = ttk.LabelFrame(cards_frame, text="Hole Cards", padding="5")
        hole_frame.pack(fill=tk.X, pady=5)

        self.hole_cards_var = tk.StringVar(value="Not detected")
        ttk.Label(
            hole_frame, textvariable=self.hole_cards_var, font=("Arial", 12)
        ).pack()

        # Community cards
        community_frame = ttk.LabelFrame(
            cards_frame, text="Community Cards", padding="5"
        )
        community_frame.pack(fill=tk.X, pady=5)

        self.community_cards_var = tk.StringVar(value="Not detected")
        ttk.Label(
            community_frame, textvariable=self.community_cards_var, font=("Arial", 12)
        ).pack()

    def create_raw_tab(self, parent: ttk.Frame) -> None:
        """Create raw data tab."""
        raw_frame = ttk.LabelFrame(parent, text="Raw Detection Data", padding="10")
        raw_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Text widget for raw data
        self.raw_text = tk.Text(raw_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(
            raw_frame, orient=tk.VERTICAL, command=self.raw_text.yview
        )
        self.raw_text.configure(yscrollcommand=scrollbar.set)

        self.raw_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def browse_image(self) -> None:
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Poker Table Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")],
        )
        if filename:
            self.image_path_var.set(filename)

    def analyze_situation(self) -> None:
        """Analyze the current poker situation."""
        image_path = self.image_path_var.get()

        if not image_path:
            messagebox.showerror("Error", "Please select an image file first.")
            return

        if not os.path.exists(image_path):
            messagebox.showerror("Error", "Selected image file does not exist.")
            return

        try:
            # Get GUI data
            gui_data = self.assistant.get_gui_data(image_path)

            if "error" in gui_data:
                messagebox.showerror("Analysis Error", gui_data["error"])
                return

            # Update AI analysis tab
            self.update_ai_tab(gui_data["ai_analysis"])

            # Update table info tab
            self.update_table_tab(gui_data["table_info"])

            # Update cards tab
            self.update_cards_tab(gui_data["cards"])

            # Update raw data tab
            self.update_raw_tab(gui_data["raw_detection"])

            messagebox.showinfo("Success", "Analysis completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze situation: {str(e)}")

    def update_ai_tab(self, ai_data: Dict[str, Any]) -> None:
        """Update AI analysis tab with new data."""
        primary = ai_data.get("primary_action", {})

        # Update recommendation
        action = primary.get("action", "unknown").upper()
        self.recommendation_var.set(action)

        # Set color based on action
        colors = {
            "FOLD": "#ff4444",
            "CALL": "#ffaa00",
            "RAISE": "#44ff44",
            "CHECK": "#4444ff",
        }
        self.recommendation_label.configure(foreground=colors.get(action, "#000000"))

        # Update confidence
        confidence = primary.get("confidence", 0.0)
        self.confidence_var.set(f"{confidence:.1%}")

        # Set confidence color
        if confidence >= 0.8:
            color = "#44ff44"  # Green
        elif confidence >= 0.6:
            color = "#ffaa00"  # Orange
        else:
            color = "#ff4444"  # Red
        self.confidence_label.configure(foreground=color)

        # Update reasoning
        reasoning = primary.get("reasoning", "No reasoning available")
        self.reasoning_text.delete(1.0, tk.END)
        self.reasoning_text.insert(1.0, reasoning)

        # Update metrics
        self.expected_value_var.set(f"${primary.get('expected_value', 0.0):.2f}")
        self.risk_level_var.set(primary.get("risk_level", "unknown").title())

        hand_info = ai_data.get("hand_info", {})
        self.hand_strength_var.set(hand_info.get("strength", "unknown").title())
        self.pot_odds_var.set(f"{hand_info.get('pot_odds', 0.0):.2f}:1")

    def update_table_tab(self, table_data: Dict[str, Any]) -> None:
        """Update table info tab with new data."""
        self.pot_size_var.set(f"${table_data.get('pot_size', 0.0):.2f}")
        self.current_bet_var.set(f"${table_data.get('current_bet', 0.0):.2f}")
        self.street_var.set(table_data.get("street", "unknown").title())

    def update_cards_tab(self, cards_data: Dict[str, Any]) -> None:
        """Update cards tab with new data."""
        hole_cards = cards_data.get("hole_cards", [])
        if hole_cards:
            self.hole_cards_var.set(" ".join(hole_cards))
        else:
            self.hole_cards_var.set("Not detected")

        community_cards = cards_data.get("community_cards", [])
        if community_cards:
            self.community_cards_var.set(" ".join(community_cards))
        else:
            self.community_cards_var.set("Not detected")

    def update_raw_tab(self, raw_data: Dict[str, Any]) -> None:
        """Update raw data tab with new data."""
        self.raw_text.delete(1.0, tk.END)
        self.raw_text.insert(1.0, json.dumps(raw_data, indent=2))


def main() -> None:
    """Main function."""
    root = tk.Tk()
    PokerAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
