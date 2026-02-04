"""
tensorboard_feedback.py - TensorBoard-Based Feedback System

Replacement for form.py that uses TensorBoard for interactive feedback collection.
Uses browser-based interface instead of Tkinter GUI.

Key Features:
- Interactive visualizations in TensorBoard dashboard
- Real-time visualization of color usage
- Structured feedback storage (JSON)
- Session comparison and history tracking
- Cross-platform compatibility (browser-based, no Tkinter)
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TensorBoardFeedbackSystem:
    """
    TensorBoard-based feedback collection for CNN color transfer.

    This class manages:
    1. TensorBoard logging of color usage statistics
    2. Visualization of color distributions
    3. Interactive feedback collection workflow
    4. Structured storage of feedback data
    """

    def __init__(self, session_name=None, log_dir='runs/feedback'):
        """
        Initialize TensorBoard feedback system.

        Args:
            session_name: Name for this feedback session (default: timestamp)
            log_dir: Base directory for TensorBoard logs
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session_name = session_name
        self.log_dir = os.path.join(log_dir, session_name)
        self.writer = SummaryWriter(self.log_dir)

        # Create feedback storage directory
        self.feedback_dir = Path('feedback_data')
        self.feedback_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print(f"TENSORBOARD FEEDBACK SESSION: {session_name}")
        print(f"{'='*70}")
        print(f"Log directory: {self.log_dir}")
        print(f"Feedback storage: {self.feedback_dir}")
        print(f"\n📊 Start TensorBoard with:")
        print(f"   tensorboard --logdir={log_dir}")
        print(f"\n🌐 Then open: http://localhost:6006")
        print(f"{'='*70}\n")

    def log_initial_visualization(self, palette_rgb, palette_lab,
                                  triangle_colors, image_original, image_cnn):
        """
        Log comprehensive initial visualization to TensorBoard.

        This creates the dashboard that users see before providing feedback.

        Args:
            palette_rgb: numpy array (10, 3) - target palette [0-255]
            palette_lab: numpy array (10, 3) - target palette in LAB
            triangle_colors: numpy array (N, 3) - CNN output for each triangle [0-255]
            image_original: PIL Image - original triangulated image
            image_cnn: PIL Image - CNN-colored triangulation
        """
        print("Logging visualizations to TensorBoard...")

        # 1. Log color palette as image
        palette_image = self._create_palette_visualization(palette_rgb, palette_lab)
        self.writer.add_image('Color_Palette/Target_Palette',
                             palette_image, 0, dataformats='HWC')

        # 2. Log color usage statistics
        self._log_color_usage_stats(palette_rgb, triangle_colors)

        # 3. Log side-by-side comparison
        comparison_image = self._create_comparison_image(image_original, image_cnn)
        self.writer.add_image('Comparison/Original_vs_CNN',
                             comparison_image, 0, dataformats='HWC')

        # 4. Log 3D color distribution
        self._log_3d_color_distribution(palette_rgb, triangle_colors)

        # 5. Log color usage heatmap
        self._log_color_usage_heatmap(palette_rgb, triangle_colors)

        print("✓ Visualizations logged successfully")

    def _create_palette_visualization(self, palette_rgb, palette_lab):
        """
        Create visual representation of color palette.

        Returns:
            numpy array (H, W, 3) suitable for TensorBoard image logging
        """
        fig, ax = plt.subplots(figsize=(12, 2))

        # Draw color swatches
        for i, (rgb, lab) in enumerate(zip(palette_rgb, palette_lab)):
            # Color swatch
            rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb/255.0,
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Label with RGB and LAB values
            label = f"Color {i+1}\nRGB{tuple(int(c) for c in rgb)}\n" \
                   f"LAB({int(lab[0])}, {int(lab[1])}, {int(lab[2])})"
            ax.text(i+0.5, -0.3, label, ha='center', va='top',
                   fontsize=8, fontfamily='monospace')

        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Target Color Palette (Lightest → Darkest)',
                    fontsize=14, fontweight='bold', pad=20)

        # Convert to numpy array for TensorBoard
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        image - np.asarray(fig.canvas.buffer_rgba())
        #image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return image

    def _log_color_usage_stats(self, palette_rgb, triangle_colors):
        """
        Log statistics about how each palette color was used.

        For each color in palette:
        - Frequency: How many triangles are closest to this color?
        - Average distance: How closely did triangles match this color?
        """
        print("  Computing color usage statistics...")

        for i, palette_color in enumerate(palette_rgb):
            # Find triangles closest to this palette color
            distances = np.linalg.norm(triangle_colors - palette_color, axis=1)

            # Frequency: percentage of triangles using this color
            frequency = (distances < 30).sum() / len(triangle_colors) * 100  # Within 30 RGB distance

            # Average distance to this color
            avg_distance = distances.mean()

            # Log to TensorBoard
            self.writer.add_scalar(f'Color_Usage/Color_{i+1:02d}_Frequency_%', frequency, 0)
            self.writer.add_scalar(f'Color_Usage/Color_{i+1:02d}_Avg_Distance', avg_distance, 0)

            # Log histogram of distances
            self.writer.add_histogram(f'Color_Distance_Dist/Color_{i+1:02d}', distances, 0)

    def _create_comparison_image(self, image_original, image_cnn):
        """
        Create side-by-side comparison of original vs CNN output.

        Returns:
            numpy array (H, W, 3) for TensorBoard
        """
        # Convert PIL images to numpy
        img1 = np.array(image_original.convert('RGB'))
        img2 = np.array(image_cnn.convert('RGB'))

        # Resize to same height if needed
        if img1.shape[0] != img2.shape[0]:
            from PIL import Image
            h = min(img1.shape[0], img2.shape[0])
            img1_pil = Image.fromarray(img1).resize((int(img1.shape[1] * h / img1.shape[0]), h))
            img2_pil = Image.fromarray(img2).resize((int(img2.shape[1] * h / img2.shape[0]), h))
            img1 = np.array(img1_pil)
            img2 = np.array(img2_pil)

        # Concatenate horizontally
        comparison = np.concatenate([img1, img2], axis=1)

        return comparison

    def _log_3d_color_distribution(self, palette_rgb, triangle_colors):
        """
        Log 3D scatter plot of color distribution in RGB space.

        Shows:
        - Gray dots: All triangle colors (CNN output)
        - Colored large dots: Target palette colors
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Downsample triangle colors for visualization
        if len(triangle_colors) > 5000:
            indices = np.random.choice(len(triangle_colors), 5000, replace=False)
            triangle_sample = triangle_colors[indices]
        else:
            triangle_sample = triangle_colors

        # Plot triangle colors (output)
        ax.scatter(triangle_sample[:, 0], triangle_sample[:, 1], triangle_sample[:, 2],
                  c=triangle_sample/255.0, s=1, alpha=0.3, label='CNN Output')

        # Plot palette colors (target)
        ax.scatter(palette_rgb[:, 0], palette_rgb[:, 1], palette_rgb[:, 2],
                  c=palette_rgb/255.0, s=200, marker='s',
                  edgecolors='black', linewidths=2, label='Target Palette')

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title('Color Distribution: CNN Output vs Target Palette')
        ax.legend()

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        self.writer.add_image('3D_Distribution/RGB_Space', image, 0, dataformats='HWC')

    def _log_color_usage_heatmap(self, palette_rgb, triangle_colors):
        """
        Create heatmap showing which palette colors are used where.

        This is a 10×H grid where:
        - Rows = 10 palette colors
        - Columns = spatial bins of triangles
        - Cell intensity = how much that color is used in that region
        """
        print("  Creating color usage heatmap...")

        fig, ax = plt.subplots(figsize=(12, 6))

        # For each palette color, compute usage frequency in bins
        num_bins = 20
        heatmap = np.zeros((10, num_bins))

        for i, palette_color in enumerate(palette_rgb):
            # Find triangles similar to this color
            distances = np.linalg.norm(triangle_colors - palette_color, axis=1)

            # Bin triangles by index (proxy for spatial distribution)
            triangle_indices = np.arange(len(triangle_colors))
            bins = np.linspace(0, len(triangle_colors), num_bins + 1)

            for j in range(num_bins):
                bin_mask = (triangle_indices >= bins[j]) & (triangle_indices < bins[j+1])
                bin_distances = distances[bin_mask]

                # Usage = percentage of triangles in bin close to this color
                if len(bin_distances) > 0:
                    heatmap[i, j] = (bin_distances < 40).sum() / len(bin_distances) * 100

        # Plot heatmap
        im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_yticks(range(10))
        ax.set_yticklabels([f'Color {i+1}' for i in range(10)])
        ax.set_xlabel('Spatial Region (Left → Right)')
        ax.set_ylabel('Palette Color (Lightest → Darkest)')
        ax.set_title('Color Usage Heatmap: Where Each Color Is Used')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Usage Frequency (%)', rotation=270, labelpad=20)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        self.writer.add_image('Usage_Heatmap/Spatial_Distribution', image, 0, dataformats='HWC')

    def collect_feedback_interactive(self, palette_rgb, palette_lab):
        """
        Collect user feedback via command-line prompts.

        Since TensorBoard doesn't support interactive sliders for data input,
        we use a simple command-line interface while user views TensorBoard dashboard.

        Args:
            palette_rgb: numpy array (10, 3)
            palette_lab: numpy array (10, 3)

        Returns:
            dict: Feedback data with scores and metadata
        """
        print("\n" + "="*70)
        print("FEEDBACK COLLECTION")
        print("="*70)
        print("\n📊 Please review the TensorBoard dashboard at:")
        print("   http://localhost:6006")
        print("\nThen rate each color below:\n")

        frequency_scores = []
        placement_scores = []

        for i in range(10):
            rgb = palette_rgb[i]
            lab = palette_lab[i]

            print(f"\n{'─'*70}")
            print(f"Color {i+1}/10 (Lightest → Darkest)")
            print(f"  RGB: {tuple(int(c) for c in rgb)}")
            print(f"  LAB: ({int(lab[0])}, {int(lab[1])}, {int(lab[2])})")
            print(f"{'─'*70}")

            # Frequency rating
            while True:
                try:
                    freq_input = input(f"Q1: Frequency (0=never used, 5=perfect, 9=use more) [default: 5]: ").strip()
                    freq_score = int(freq_input) if freq_input else 5
                    if 0 <= freq_score <= 9:
                        frequency_scores.append(freq_score)
                        break
                    else:
                        print("  ⚠️  Please enter a number between 0 and 9")
                except ValueError:
                    print("  ⚠️  Please enter a valid number")

            # Placement rating (only if frequency > 0)
            if freq_score > 0:
                while True:
                    try:
                        place_input = input(f"Q2: Placement (0=wrong places, 5=good, 9=excellent) [default: 5]: ").strip()
                        place_score = int(place_input) if place_input else 5
                        if 0 <= place_score <= 9:
                            placement_scores.append(place_score)
                            break
                        else:
                            print("  ⚠️  Please enter a number between 0 and 9")
                    except ValueError:
                        print("  ⚠️  Please enter a valid number")
            else:
                placement_scores.append(0)
                print("  ℹ️  Placement set to 0 (frequency is 0)")

        # Create feedback data structure
        feedback_data = {
            'session_name': self.session_name,
            'timestamp': datetime.now().isoformat(),
            'frequency_scores': frequency_scores,
            'placement_scores': placement_scores,
            'palette_rgb': palette_rgb.tolist(),
            'palette_lab': palette_lab.tolist()
        }

        # Log feedback scores to TensorBoard
        self._log_feedback_scores(frequency_scores, placement_scores)

        # Save feedback to JSON
        self._save_feedback_json(feedback_data)

        print("\n" + "="*70)
        print("✓ FEEDBACK COLLECTED SUCCESSFULLY")
        print("="*70)
        print(f"Frequency scores: {frequency_scores}")
        print(f"Placement scores: {placement_scores}")
        print("="*70 + "\n")

        return feedback_data

    def _log_feedback_scores(self, frequency_scores, placement_scores):
        """Log feedback scores as TensorBoard scalars."""
        for i in range(10):
            self.writer.add_scalar(f'Feedback/Color_{i+1:02d}_Frequency',
                                  frequency_scores[i], 0)
            self.writer.add_scalar(f'Feedback/Color_{i+1:02d}_Placement',
                                  placement_scores[i], 0)

    def _save_feedback_json(self, feedback_data):
        """Save feedback to JSON file."""
        filename = self.feedback_dir / f"{self.session_name}.json"
        with open(filename, 'w') as f:
            json.dump(feedback_data, f, indent=2)

        print(f"💾 Feedback saved to: {filename}")

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
        print(f"✓ TensorBoard session closed: {self.session_name}")


def load_all_feedback(feedback_dir='feedback_data'):
    """
    Load all previous feedback sessions from JSON files.

    Args:
        feedback_dir: Directory containing feedback JSON files

    Returns:
        list: List of feedback dictionaries, sorted by timestamp (oldest first)

    Example:
        all_feedback = load_all_feedback()
        for session in all_feedback:
            print(f"Session: {session['session_name']}")
            print(f"  Frequency: {session['frequency_scores']}")
            print(f"  Placement: {session['placement_scores']}")
    """
    feedback_dir = Path(feedback_dir)

    if not feedback_dir.exists():
        return []

    feedback_list = []

    for json_file in feedback_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                feedback_data = json.load(f)
                feedback_list.append(feedback_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Warning: Could not load {json_file}: {e}")
            continue

    # Sort by timestamp (oldest first)
    feedback_list.sort(key=lambda x: x.get('timestamp', ''))

    return feedback_list


def get_user_feedback_tensorboard(palette_rgb, palette_lab, triangle_colors,
                                  image_original, image_cnn, session_name=None):
    """
    Complete TensorBoard-based feedback workflow.

    This is the main entry point that replaces form.get_user_feedback().

    Args:
        palette_rgb: numpy array (10, 3) - target palette [0-255]
        palette_lab: numpy array (10, 3) - target palette in LAB
        triangle_colors: numpy array (N, 3) - CNN output colors [0-255]
        image_original: PIL Image - original triangulation
        image_cnn: PIL Image - CNN-colored triangulation
        session_name: Optional name for this session

    Returns:
        list: [freq0, ..., freq9, place0, ..., place9] (same format as form.py)

    Example:
        scores = get_user_feedback_tensorboard(
            palette_rgb, palette_lab, triangle_colors,
            image_original, image_cnn
        )
        # Returns: [5,7,3,9,8,2,6,4,1,0,5,7,3,9,8,2,6,4,1,0]
    """
    # Create feedback system
    tb_feedback = TensorBoardFeedbackSystem(session_name=session_name)

    # Log visualizations
    tb_feedback.log_initial_visualization(
        palette_rgb, palette_lab, triangle_colors,
        image_original, image_cnn
    )

    # Collect feedback interactively
    feedback_data = tb_feedback.collect_feedback_interactive(palette_rgb, palette_lab)

    # Close TensorBoard writer
    tb_feedback.close()

    # Return in same format as form.py (for compatibility)
    scores = feedback_data['frequency_scores'] + feedback_data['placement_scores']
    return scores


def scores_to_filename_suffix(scores):
    """
    Convert scores list to filename suffix (for backward compatibility).

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
    # Test the system with dummy data
    print("Testing TensorBoard Feedback System...")

    # Create dummy palette
    test_palette_rgb = np.random.randint(50, 200, (10, 3)).astype(np.float32)
    test_palette_lab = np.random.randint(20, 100, (10, 3)).astype(np.float32)

    # Create dummy triangle colors
    test_triangle_colors = np.random.randint(0, 255, (1000, 3)).astype(np.float32)

    # Create dummy images (placeholder)
    from PIL import Image
    test_image_original = Image.new('RGB', (400, 400), color='white')
    test_image_cnn = Image.new('RGB', (400, 400), color='lightgray')

    # Run feedback collection
    scores = get_user_feedback_tensorboard(
        test_palette_rgb, test_palette_lab, test_triangle_colors,
        test_image_original, test_image_cnn,
        session_name='test_session'
    )

    print(f"\n✓ Test complete! Scores: {scores}")
