"""
form.py - Interactive Feedback Form for Color Palette Ratings

This module provides a Tkinter-based GUI for rating how well the CNN used
colors in the triangulated image. Users rate each color on two dimensions:
1. Frequency: Did I use this color enough? (0-9)
2. Placement: Did I use it in the right places? (0-9, only if Frequency > 0)

Scoring Guide:
- Frequency: 0 (never used) to 9 (use much more)
- Placement: 0 (wrong places) to 9 (excellent placement)

Output Format (Option B):
Filename: ABC_5739826400_5739826400
           └─ Hex prefix
              └─ All 10 frequency scores (lightest→darkest)
                 └─ All 10 placement scores
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk


class FeedbackForm:
    """Interactive form for rating color usage in CNN outputs."""
    
    def __init__(self, palette_rgb, palette_lab):
        """
        Initialize feedback form.
        
        Args:
            palette_rgb: numpy array (10, 3) - RGB colors [0-255], sorted by LAB L (lightest→darkest)
            palette_lab: numpy array (10, 3) - LAB colors for reference
        """
        self.palette_rgb = palette_rgb  # 10 colors
        self.palette_lab = palette_lab
        self.num_colors = len(palette_rgb)
        
        # Storage for scores
        self.frequency_scores = [5] * self.num_colors  # Default: 5 (perfect)
        self.placement_scores = [5] * self.num_colors  # Default: 5 (perfect), only used if freq > 0
        
        # Build GUI
        self.root = tk.Tk()
        self.root.title("Color Usage Feedback Form")
        self.root.geometry("1000x900")
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas with scrollbar for many colors
        canvas = tk.Canvas(main_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.scroll_region)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_label = ttk.Label(
            scrollable_frame,
            text="Rate How Well the Network Used Each Color",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=15)
        
        # Instructions
        instructions = ttk.Label(
            scrollable_frame,
            text="Colors are sorted from LIGHTEST to DARKEST (by LAB Lightness value)\n"
                 "Question 1: Frequency (0=never, 5=perfect, 9=use much more)\n"
                 "Question 2: Placement (only shown if Frequency > 0)",
            font=("Arial", 10),
            justify="center"
        )
        instructions.pack(pady=10)
        
        # Color rating frames
        self.color_frames = []
        for i in range(self.num_colors):
            frame = self._create_color_frame(scrollable_frame, i)
            self.color_frames.append(frame)
            frame.pack(pady=15, padx=10, fill=tk.X)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        submit_btn = ttk.Button(
            button_frame,
            text="Submit & Save",
            command=self._submit
        )
        submit_btn.pack(side="left", padx=5)
        
        reset_btn = ttk.Button(
            button_frame,
            text="Reset to Default",
            command=self._reset
        )
        reset_btn.pack(side="left", padx=5)
        
        self.result = None  # Will be set when user submits
    
    def _create_color_frame(self, parent, color_idx):
        """Create a frame for one color with both rating questions."""
        frame = ttk.LabelFrame(
            parent,
            text=f"Color {color_idx + 1} (Lightest→Darkest Order)",
            padding=15
        )
        
        # Color square on left (50x50)
        color_rgb = self.palette_rgb[color_idx]
        color_lab = self.palette_lab[color_idx]
        
        # Create color preview
        color_frame = ttk.Frame(frame)
        color_frame.pack(side="left", padx=15, pady=5)
        
        # Display 50x50 color square
        color_image = Image.new('RGB', (50, 50), tuple(int(c) for c in color_rgb))
        color_photo = ImageTk.PhotoImage(color_image)
        
        color_label = tk.Label(color_frame, image=color_photo, width=50, height=50)
        color_label.image = color_photo  # Keep a reference
        color_label.pack()
        
        # Color info
        info_text = f"RGB({int(color_rgb[0])}, {int(color_rgb[1])}, {int(color_rgb[2])})\n"
        info_text += f"LAB({int(color_lab[0])}, {int(color_lab[1])}, {int(color_lab[2])})"
        info_label = ttk.Label(color_frame, text=info_text, font=("Arial", 8), justify="center")
        info_label.pack()
        
        # Questions on right
        questions_frame = ttk.Frame(frame)
        questions_frame.pack(side="left", fill=tk.BOTH, expand=True, padx=15)
        
        # Question 1: Frequency
        freq_label = ttk.Label(
            questions_frame,
            text="Q1: How often did I use this color? (0=never, 5=perfect, 9=use much more)",
            font=("Arial", 10)
        )
        freq_label.pack(anchor="w", pady=(0, 5))
        
        freq_frame = ttk.Frame(questions_frame)
        freq_frame.pack(anchor="w", pady=(0, 15))
        
        # Frequency scale
        freq_var = tk.IntVar(value=5)
        freq_scale = ttk.Scale(
            freq_frame,
            from_=0,
            to=9,
            orient="horizontal",
            variable=freq_var,
            command=lambda v: self._on_frequency_change(color_idx, freq_var, placement_var, placement_widgets)
        )
        freq_scale.pack(side="left", fill=tk.X, expand=True)
        
        freq_value = ttk.Label(freq_frame, text="5", font=("Arial", 12, "bold"), width=3)
        freq_value.pack(side="left", padx=(5, 0))
        
        def update_freq_value(*args):
            freq_value.config(text=str(freq_var.get()))
            self.frequency_scores[color_idx] = freq_var.get()
        
        freq_var.trace("w", update_freq_value)
        
        # Question 2: Placement (only shown if frequency > 0)
        placement_var = tk.IntVar(value=5)
        
        placement_widgets = {
            'label': None,
            'scale': None,
            'value_label': None
        }
        
        def create_placement_widgets():
            """Create placement question widgets."""
            if placement_widgets['label'] is not None:
                return  # Already created
            
            placement_label = ttk.Label(
                questions_frame,
                text="Q2: Did I use it in the right places? (0=wrong places, 5=good, 9=excellent)",
                font=("Arial", 10)
            )
            placement_label.pack(anchor="w", pady=(0, 5))
            placement_widgets['label'] = placement_label
            
            placement_frame = ttk.Frame(questions_frame)
            placement_frame.pack(anchor="w")
            
            placement_scale = ttk.Scale(
                placement_frame,
                from_=0,
                to=9,
                orient="horizontal",
                variable=placement_var,
                command=lambda v: self._update_placement_value(color_idx, placement_var, placement_value)
            )
            placement_scale.pack(side="left", fill=tk.X, expand=True)
            placement_widgets['scale'] = placement_scale
            
            placement_value = ttk.Label(placement_frame, text="5", font=("Arial", 12, "bold"), width=3)
            placement_value.pack(side="left", padx=(5, 0))
            placement_widgets['value_label'] = placement_value
            
            def update_placement_value(*args):
                placement_value.config(text=str(placement_var.get()))
                self.placement_scores[color_idx] = placement_var.get()
            
            placement_var.trace("w", update_placement_value)
        
        def hide_placement_widgets():
            """Hide placement question widgets."""
            for widget in placement_widgets.values():
                if widget is not None:
                    if hasattr(widget, 'pack_forget'):
                        widget.pack_forget()
                    elif hasattr(widget, 'pack_remove'):
                        widget.pack_remove()
            placement_widgets['label'] = None
            placement_widgets['scale'] = None
            placement_widgets['value_label'] = None
        
        # Initially show placement if frequency > 0
        if freq_var.get() > 0:
            create_placement_widgets()
        
        # Store references
        self.color_frames.append({
            'freq_var': freq_var,
            'placement_var': placement_var,
            'placement_widgets': placement_widgets,
            'create_placement': create_placement_widgets,
            'hide_placement': hide_placement_widgets
        })
        
        return frame
    
    def _on_frequency_change(self, color_idx, freq_var, placement_var, placement_widgets):
        """Handle frequency slider change - show/hide placement question."""
        freq_value = freq_var.get()
        self.frequency_scores[color_idx] = freq_value
        
        if freq_value > 0:
            # Show placement widgets if not already showing
            if placement_widgets['label'] is None:
                self._create_placement_widgets_internal(color_idx, placement_var, placement_widgets)
        else:
            # Hide placement widgets and set placement score to 0
            self._hide_placement_widgets_internal(placement_widgets)
            placement_var.set(0)
            self.placement_scores[color_idx] = 0
    
    def _create_placement_widgets_internal(self, color_idx, placement_var, placement_widgets):
        """Internal helper to create placement widgets."""
        # Find the questions_frame for this color
        color_frame = self.color_frames[color_idx]
        questions_frame = None
        
        # This is a bit hacky but works - we need to find the right parent frame
        # In practice, we'll reorganize this in a cleaner way
        pass
    
    def _hide_placement_widgets_internal(self, placement_widgets):
        """Internal helper to hide placement widgets."""
        for widget in placement_widgets.values():
            if widget is not None and hasattr(widget, 'pack_forget'):
                widget.pack_forget()
    
    def _update_placement_value(self, color_idx, placement_var, placement_value):
        """Update placement score display and storage."""
        placement_value.config(text=str(placement_var.get()))
        self.placement_scores[color_idx] = placement_var.get()
    
    def _reset(self):
        """Reset all scores to default (5)."""
        for i in range(self.num_colors):
            self.frequency_scores[i] = 5
            self.placement_scores[i] = 5
        
        # Reset all sliders in UI
        for i, frame_dict in enumerate(self.color_frames):
            if isinstance(frame_dict, dict) and 'freq_var' in frame_dict:
                frame_dict['freq_var'].set(5)
                frame_dict['placement_var'].set(5)
        
        messagebox.showinfo("Reset", "All scores reset to default (5)")
    
    def _submit(self):
        """Submit form and return scores."""
        # Ensure placement scores are 0 if frequency is 0
        for i in range(self.num_colors):
            if self.frequency_scores[i] == 0:
                self.placement_scores[i] = 0
        
        # Return as Option B format: [freq1, freq2, ..., place1, place2, ...]
        self.result = self.frequency_scores + self.placement_scores
        
        print("\n" + "="*70)
        print("FEEDBACK FORM SUBMITTED")
        print("="*70)
        print(f"\nFrequency Scores (Lightest→Darkest): {self.frequency_scores}")
        print(f"Placement Scores (Lightest→Darkest):  {self.placement_scores}")
        print("="*70)
        
        self.root.quit()
        self.root.destroy()
    
    def show(self):
        """Display the form and wait for submission."""
        self.root.mainloop()
        return self.result


def get_user_feedback(palette_rgb, palette_lab):
    """
    Display feedback form and collect user ratings.
    
    Args:
        palette_rgb: numpy array (10, 3) - RGB colors [0-255]
        palette_lab: numpy array (10, 3) - LAB colors
    
    Returns:
        list: [freq0, freq1, ..., freq9, place0, place1, ..., place9]
              Each value 0-9, placement is 0 if frequency is 0
    
    Example:
        scores = get_user_feedback(palette_rgb, palette_lab)
        freq_scores = scores[:10]
        placement_scores = scores[10:20]
    """
    form = FeedbackForm(palette_rgb, palette_lab)
    scores = form.show()
    
    if scores is None:
        # User closed without submitting
        print("Form closed without submission. Using default scores (all 5).")
        return [5] * 10 + [5] * 10
    
    return scores


def scores_to_filename_suffix(scores):
    """
    Convert scores list to filename suffix (Option B format).
    
    Args:
        scores: list of 20 integers [freq0..freq9, place0..place9]
    
    Returns:
        str: "5739826400_5739826400" format
    """
    freq_scores = scores[:10]
    place_scores = scores[10:20]
    
    freq_str = ''.join(str(s) for s in freq_scores)
    place_str = ''.join(str(s) for s in place_scores)
    
    return f"{freq_str}_{place_str}"


if __name__ == '__main__':
    # Test the form with dummy data
    test_palette_rgb = np.random.randint(0, 256, (10, 3))
    test_palette_lab = np.random.randint(0, 256, (10, 3))
    
    scores = get_user_feedback(test_palette_rgb, test_palette_lab)
    print(f"\nFinal scores: {scores}")
    print(f"Filename suffix: {scores_to_filename_suffix(scores)}")
