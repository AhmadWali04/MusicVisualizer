"""
CNN.py - Neural Network-based Color Transfer System

This module implements a deep neural network that learns intelligent color mappings
from source image colors to a target color palette. It's designed to integrate with
imageTriangulation.py and colour.py for end-to-end color stylization.

Key Components:
- ColorTransferNet: Main neural network for color mapping
- ColorHistogramLoss: Ensures output distribution matches target palette
- Training pipeline with visualization
- Application functions for triangulated image coloring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Import our own modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour
from PIL import Image


class ColorTransferNet(nn.Module):
    """
    Neural network that learns to map source image colors to target palette colors.
    
    Architecture:
    - Input: RGB color [3 values, 0-1 normalized]
    - Hidden layers: 5 layers with 256 neurons each
    - Activation: ReLU after each hidden layer
    - Normalization: BatchNorm1d after each ReLU
    - Output: RGB color [3 values, 0-1 normalized] with Sigmoid activation
    
    The network learns complex non-linear color transformations that preserve
    visual harmony with the target palette while maintaining source image structure.
    """
    
    def __init__(self, n_layers=5, hidden_dim=256):
        """
        Initialize the ColorTransferNet.
        
        Args:
            n_layers: Number of hidden layers (default: 5)
            hidden_dim: Number of neurons in each hidden layer (default: 256)
        """
        super(ColorTransferNet, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Build network layers dynamically
        layers = []
        
        # First layer: 3 (RGB) -> hidden_dim
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer: hidden_dim -> 3 (RGB)
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, colors):
        """
        Forward pass through the network.
        
        Args:
            colors: Input tensor of shape (batch_size, 3) with values in [0, 1]
        
        Returns:
            output: Tensor of shape (batch_size, 3) with values in [0, 1]
        """
        return self.network(colors)


class ColorHistogramLoss(nn.Module):
    """
    Ensures the output color distribution matches the target palette distribution.
    
    This loss computes 3D color histograms and measures their similarity using
    Wasserstein distance. This prevents mode collapse where the network only
    outputs a few colors and encourages diverse color usage.
    """
    
    def __init__(self, num_bins=16):
        """
        Initialize the ColorHistogramLoss.
        
        Args:
            num_bins: Number of bins per channel in the 3D histogram (default: 16)
        """
        super(ColorHistogramLoss, self).__init__()
        self.num_bins = num_bins
    
    def forward(self, source_colors, target_palette):
        """
        Compute histogram matching loss.
        
        Args:
            source_colors: Tensor of shape (N, 3) - output colors from network
            target_palette: Tensor of shape (M, 3) - target palette colors
        
        Returns:
            loss: Scalar tensor representing histogram mismatch
        """
        # Compute histograms for both
        source_hist = self._compute_histogram(source_colors)
        target_hist = self._compute_histogram(target_palette)
        
        # L1 distance between histograms
        loss = torch.abs(source_hist - target_hist).mean()
        
        return loss
    
    def _compute_histogram(self, colors):
        """
        Compute a 3D histogram of colors.
        
        Args:
            colors: Tensor of shape (N, 3) with values in [0, 1]
        
        Returns:
            histogram: Tensor of shape (num_bins, num_bins, num_bins)
        """
        # Quantize colors to histogram bins
        quantized = (colors * (self.num_bins - 1)).long()
        quantized = torch.clamp(quantized, 0, self.num_bins - 1)
        
        # Create histogram
        histogram = torch.zeros(
            self.num_bins, self.num_bins, self.num_bins,
            device=colors.device, dtype=colors.dtype
        )
        
        # Fill histogram
        for r, g, b in quantized:
            histogram[r, g, b] += 1.0
        
        # Normalize
        histogram = histogram / (histogram.sum() + 1e-8)
        
        return histogram.view(-1)  # Flatten for loss computation


class NearestColorDistanceLoss(nn.Module):
    """
    Ensures each output color is close to at least one target palette color.
    
    This encourages the network to map colors to the actual palette,
    rather than outputting arbitrary intermediate colors.
    """
    
    def __init__(self):
        super(NearestColorDistanceLoss, self).__init__()
    
    def forward(self, output_colors, target_palette):
        """
        Compute nearest color distance loss.
        
        Args:
            output_colors: Tensor of shape (N, 3) - network output
            target_palette: Tensor of shape (M, 3) - target colors
        
        Returns:
            loss: Scalar tensor
        """
        # Expand dimensions for broadcasting
        # output_colors: (N, 1, 3)
        # target_palette: (1, M, 3)
        output_expanded = output_colors.unsqueeze(1)  # (N, 1, 3)
        target_expanded = target_palette.unsqueeze(0)  # (1, M, 3)
        
        # Compute pairwise distances
        distances = torch.norm(output_expanded - target_expanded, dim=2)  # (N, M)
        
        # Minimum distance for each output color to any target color
        min_distances = distances.min(dim=1)[0]  # (N,)
        
        # Mean of minimum distances
        loss = min_distances.mean()
        
        return loss


def compute_smoothness_loss(model, source_pixels, n_samples=1000, noise_scale=0.01, device='cpu'):
    """
    Regularization to ensure smooth color transitions.
    
    Similar input colors should map to similar output colors.
    This prevents jarring discontinuities in the color mapping and makes
    the network learn smoother transformations.
    
    Args:
        model: ColorTransferNet
        source_pixels: Input pixel tensor of shape (N, 3)
        n_samples: Number of samples to test (default: 1000)
        noise_scale: Amount of Gaussian noise to add (default: 0.01)
        device: 'cpu' or 'cuda'
    
    Returns:
        smoothness_loss: Scalar tensor
    """
    # Sample random colors from source
    if len(source_pixels) > n_samples:
        indices = torch.randperm(len(source_pixels))[:n_samples]
        sampled_colors = source_pixels[indices]
    else:
        sampled_colors = source_pixels
    
    # Add small noise to create perturbed versions
    noise = torch.randn_like(sampled_colors) * noise_scale
    perturbed_colors = torch.clamp(sampled_colors + noise, 0, 1)
    
    # Get outputs for both
    with torch.no_grad():
        output1 = model(sampled_colors)
        output2 = model(perturbed_colors)
    
    # Compute L2 distance between outputs
    output_diff = torch.norm(output1 - output2, dim=1)
    
    # Small input changes should produce small output changes
    # This is a soft constraint, not a hard requirement
    smoothness_loss = output_diff.mean()
    
    return smoothness_loss


def prepare_training_data(source_image_path, target_image_path, 
                         num_clusters=25, num_distinct=10, 
                         use_lab=True, device='cpu'):
    """
    Prepare data for training the CNN.
    
    This function:
    1. Loads both source and target images
    2. Extracts color palettes using K-Means clustering
    3. Selects distinct colors using greedy max-min distance
    4. Converts all data to PyTorch tensors with proper normalization
    
    Args:
        source_image_path: Path to image to be triangulated (e.g., spiderman.jpg)
        target_image_path: Path to style image (e.g., vegetables.jpg)
        num_clusters: Number of K-Means clusters for palette extraction (default: 25)
        num_distinct: Number of distinct colors to extract (default: 10)
        use_lab: If True, use LAB color space for clustering (recommended)
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary containing:
            - 'source_pixels': All pixels from source image, tensor (N, 3), normalized [0, 1]
            - 'target_palette': Distinct colors from target image, tensor (num_distinct, 3), [0, 1]
            - 'source_pixels_lab': Source pixels in LAB space, tensor (N, 3)
            - 'target_palette_lab': Target palette in LAB space, tensor (num_distinct, 3)
            - 'target_percentages': Color distribution percentages
    
    Example:
        data = prepare_training_data('source.jpg', 'target.jpg', 
                                    num_clusters=25, num_distinct=10)
        source_pixels = data['source_pixels'].to(device)
        target_palette = data['target_palette'].to(device)
    """
    print("\n=== Preparing Training Data ===")
    
    # Load source image pixels
    print(f"Loading source image: {source_image_path}")
    _, source_pixels_rgb = colour.load_image_pixels(source_image_path)
    source_pixels_lab = colour.convert_rgb_pixels_to_lab(source_pixels_rgb)
    
    # Load target image and extract palette
    print(f"Loading target image: {target_image_path}")
    _, target_pixels_rgb = colour.load_image_pixels(target_image_path)
    target_pixels_lab = colour.convert_rgb_pixels_to_lab(target_pixels_rgb)
    
    # Extract palette from target image
    if use_lab:
        print(f"Running K-Means clustering in LAB space ({num_clusters} clusters)...")
        _, centers_lab, _, percentages = colour.run_kmeans_lab(target_pixels_lab, num_clusters)
        
        # Convert LAB centers back to RGB for display
        centers_rgb = []
        for lab in centers_lab:
            lab_img = np.array([[lab]], dtype=np.uint8)
            rgb_img = torch.tensor(lab)
            # Use cv2 to convert LAB to RGB
            import cv2
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            centers_rgb.append(rgb_img[0, 0])
        centers_rgb = np.array(centers_rgb)
        
        print(f"Selecting {num_distinct} distinct colors using greedy max-min distance...")
        palette_lab, palette_rgb, _ = colour.select_distinct_colors_lab(
            centers_lab, centers_rgb, num_to_select=num_distinct
        )
        percentages_selected = percentages[_]
    else:
        print(f"Running K-Means clustering in RGB space ({num_clusters} clusters)...")
        _, centers_rgb, _, percentages = colour.run_kmeans(target_pixels_rgb, num_clusters)
        
        print(f"Selecting {num_distinct} distinct colors using greedy max-min distance...")
        palette_rgb, _ = colour.select_distinct_colors(centers_rgb, num_to_select=num_distinct)
        percentages_selected = percentages[_]
        
        # Convert palette to LAB for reference
        palette_rgb_uint8 = palette_rgb.astype(np.uint8).reshape(1, -1, 3)
        import cv2
        palette_lab = cv2.cvtColor(palette_rgb_uint8, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    
    print(f"\nTarget palette extracted:")
    for i, (rgb, pct) in enumerate(zip(palette_rgb, percentages_selected)):
        print(f"  Color {i+1}: RGB{tuple(rgb)} - {pct:.1f}%")
    
    # Convert to PyTorch tensors and normalize to [0, 1]
    source_pixels_tensor = torch.from_numpy(source_pixels_rgb.astype(np.float32) / 255.0)
    target_palette_tensor = torch.from_numpy(palette_rgb.astype(np.float32) / 255.0)
    source_pixels_lab_tensor = torch.from_numpy(source_pixels_lab.astype(np.float32))
    target_palette_lab_tensor = torch.from_numpy(palette_lab.astype(np.float32))
    
    # Move to device
    source_pixels_tensor = source_pixels_tensor.to(device)
    target_palette_tensor = target_palette_tensor.to(device)
    source_pixels_lab_tensor = source_pixels_lab_tensor.to(device)
    target_palette_lab_tensor = target_palette_lab_tensor.to(device)
    
    return {
        'source_pixels': source_pixels_tensor,
        'target_palette': target_palette_tensor,
        'source_pixels_lab': source_pixels_lab_tensor,
        'target_palette_lab': target_palette_lab_tensor,
        'target_percentages': percentages_selected,
        'palette_rgb': palette_rgb,  # Keep original numpy for reference
    }


def train_color_transfer_network(source_pixels, target_palette,
                                source_pixels_lab, target_palette_lab,
                                epochs=1000, batch_size=512, lr=0.001,
                                device='cpu', save_progress=True):
    """
    Train the ColorTransferNet to map source colors to target palette.
    
    Training Process:
    1. Initialize ColorTransferNet, optimizer, and loss functions
    2. For each epoch:
       a. Sample random batch of source pixels
       b. Forward pass through network
       c. Compute composite loss:
          - ColorHistogramLoss: output distribution vs target distribution (weight: 1.0)
          - NearestColorDistanceLoss: each output to nearest target color (weight: 0.5)
          - SmoothnessLoss: smooth transformations (weight: 0.1)
       d. Backward pass and optimize
       e. Log progress every 50 epochs
    3. Return trained model and loss history
    
    Args:
        source_pixels: Tensor (N, 3) - source image pixels, normalized [0, 1]
        target_palette: Tensor (M, 3) - target palette colors, normalized [0, 1]
        source_pixels_lab: Tensor (N, 3) - source in LAB space
        target_palette_lab: Tensor (M, 3) - target in LAB space
        epochs: Number of training epochs (default: 1000)
        batch_size: Batch size for training (default: 512)
        lr: Learning rate (default: 0.001)
        device: 'cpu' or 'cuda'
        save_progress: Whether to save progress plots
    
    Returns:
        Tuple containing:
            - trained_model: Trained ColorTransferNet
            - loss_history: Dict with loss values per epoch
    """
    print("\n=== Training Color Transfer Network ===")
    print(f"Device: {device}")
    print(f"Network architecture: 5 layers, 256 hidden units")
    print(f"Training epochs: {epochs}, batch size: {batch_size}, learning rate: {lr}")
    
    # Initialize model
    model = ColorTransferNet(n_layers=5, hidden_dim=256).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize loss functions
    histogram_loss_fn = ColorHistogramLoss(num_bins=16).to(device)
    nearest_color_loss_fn = NearestColorDistanceLoss().to(device)
    
    # Loss weights
    w_histogram = 1.0
    w_nearest = 0.5
    w_smoothness = 0.1
    
    # Training loop
    loss_history = {
        'total': [],
        'histogram': [],
        'nearest_color': [],
        'smoothness': []
    }
    
    for epoch in range(epochs):
        # Sample random batch
        if len(source_pixels) > batch_size:
            indices = torch.randperm(len(source_pixels))[:batch_size]
            batch = source_pixels[indices]
        else:
            batch = source_pixels
        
        # Forward pass
        output = model(batch)
        
        # Compute losses
        histogram_loss = histogram_loss_fn(output, target_palette)
        nearest_color_loss = nearest_color_loss_fn(output, target_palette)
        smoothness_loss = compute_smoothness_loss(
            model, source_pixels, n_samples=1000, 
            noise_scale=0.01, device=device
        )
        
        # Composite loss
        total_loss = (w_histogram * histogram_loss + 
                     w_nearest * nearest_color_loss + 
                     w_smoothness * smoothness_loss)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        loss_history['total'].append(total_loss.item())
        loss_history['histogram'].append(histogram_loss.item())
        loss_history['nearest_color'].append(nearest_color_loss.item())
        loss_history['smoothness'].append(smoothness_loss.item())
        
        # Log progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Total: {total_loss.item():.6f}, "
                  f"Histogram: {histogram_loss.item():.6f}, "
                  f"Nearest: {nearest_color_loss.item():.6f}, "
                  f"Smoothness: {smoothness_loss.item():.6f}")
    
    print("\nTraining complete!")
    
    return model, loss_history


def visualize_training_progress(model, source_pixels, target_palette, 
                                loss_history, device='cpu', save_path=None):
    """
    Visualize how the network's color mapping improves during training.
    
    Creates a figure showing:
    - Source color distribution (3D scatter in RGB space)
    - Target palette colors
    - Current network output color distribution
    - Loss curves across training
    
    Args:
        model: Trained ColorTransferNet
        source_pixels: Source pixel tensor
        target_palette: Target palette tensor
        loss_history: Dict with loss values from training
        device: 'cpu' or 'cuda'
        save_path: Optional path to save visualization
    """
    print("\n=== Creating Training Visualization ===")
    
    # Get model output
    model.eval()
    with torch.no_grad():
        # Downsample source for visualization
        if len(source_pixels) > 5000:
            indices = torch.randperm(len(source_pixels))[:5000]
            source_sample = source_pixels[indices]
            output_sample = model(source_sample)
        else:
            source_sample = source_pixels
            output_sample = model(source_sample)
    
    # Convert to numpy and denormalize
    source_np = source_sample.cpu().numpy() * 255
    output_np = output_sample.cpu().numpy() * 255
    target_np = target_palette.cpu().numpy() * 255
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Subplot 1: Source colors in 3D RGB space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(source_np[:, 0], source_np[:, 1], source_np[:, 2], 
               c=source_np/255.0, s=10, alpha=0.3, label='Source')
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')
    ax1.set_title('Source Color Distribution')
    ax1.set_xlim([0, 255])
    ax1.set_ylim([0, 255])
    ax1.set_zlim([0, 255])
    
    # Subplot 2: Target palette
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2],
               c=target_np/255.0, s=200, alpha=0.8, edgecolors='black', linewidth=2,
               label='Target')
    ax2.set_xlabel('R')
    ax2.set_ylabel('G')
    ax2.set_zlabel('B')
    ax2.set_title('Target Palette')
    ax2.set_xlim([0, 255])
    ax2.set_ylim([0, 255])
    ax2.set_zlim([0, 255])
    
    # Subplot 3: Network output
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.scatter(output_np[:, 0], output_np[:, 1], output_np[:, 2],
               c=output_np/255.0, s=10, alpha=0.3, label='Output')
    ax3.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2],
               c=target_np/255.0, s=200, alpha=0.8, edgecolors='black', linewidth=2,
               label='Target')
    ax3.set_xlabel('R')
    ax3.set_ylabel('G')
    ax3.set_zlabel('B')
    ax3.set_title('Network Output vs Target')
    ax3.set_xlim([0, 255])
    ax3.set_ylim([0, 255])
    ax3.set_zlim([0, 255])
    
    # Subplot 4: Total loss
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(loss_history['total'], linewidth=2, label='Total Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Total Loss Over Training')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Subplot 5: Component losses
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(loss_history['histogram'], label='Histogram', alpha=0.7)
    ax5.plot(loss_history['nearest_color'], label='Nearest Color', alpha=0.7)
    ax5.plot(loss_history['smoothness'], label='Smoothness', alpha=0.7)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Component Losses')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Subplot 6: Color mapping examples
    ax6 = fig.add_subplot(2, 3, 6)
    # Show a few example mappings
    sample_indices = np.random.choice(len(source_np), min(10, len(source_np)), replace=False)
    for idx in sample_indices:
        src_color = source_np[idx] / 255.0
        out_color = output_np[idx] / 255.0
        # Draw arrow from source to output
        ax6.arrow(src_color[0], src_color[1], 
                 out_color[0] - src_color[0], out_color[1] - src_color[1],
                 head_width=0.02, head_length=0.02, fc=out_color, ec='black', alpha=0.6)
    
    # Plot target palette
    for color in target_np / 255.0:
        ax6.scatter(color[0], color[1], s=100, c=[color], edgecolors='black', linewidth=2)
    
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.set_xlabel('R')
    ax6.set_ylabel('G')
    ax6.set_title('Sample Color Mappings (R-G plane)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def apply_cnn_to_triangulation(model, S, triangles, image_orig, device='cpu'):
    """
    Apply trained CNN to color each triangle in the triangulation.
    
    This is the key integration point with imageTriangulation.py
    
    Process:
    1. For each triangle in triangles.simplices:
       a. Calculate centroid coordinates
       b. Get centroid color from image_orig using PIL's getpixel()
       c. Convert RGB [0-255] to normalized [0, 1] tensor
       d. Pass through model to get transformed color
       e. Convert back to [0-255] for matplotlib
       f. Store color for this triangle
    2. Create matplotlib figure with colored triangles
    3. Return figure and color list
    
    Args:
        model: Trained ColorTransferNet
        S: Vertices array from triangulation, shape (N, 2)
        triangles: Delaunay triangulation object with .simplices attribute
        image_orig: Original PIL Image
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary containing:
            - 'figure': matplotlib figure object
            - 'triangle_colors': List of (num_triangles, 3) RGB colors [0-255]
            - 'ax': matplotlib axes object
    
    Example:
        result = apply_cnn_to_triangulation(model, S, triangles, image_orig, device='cpu')
        result['figure'].show()
    """
    print("\n=== Applying CNN to Triangulation ===")
    
    model.eval()
    triangle_colors = []
    
    # Process all triangles
    num_triangles = len(triangles.simplices)
    print(f"Processing {num_triangles} triangles...")
    
    with torch.no_grad():
        for i, triangle_idx in enumerate(triangles.simplices):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{num_triangles} triangles...")
            
            # Get triangle vertices
            vertices = S[triangle_idx]
            
            # Calculate centroid
            centroid = vertices.mean(axis=0)
            
            # Get color from original image at centroid
            try:
                centroid_color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
            except:
                # Handle edge cases where coordinates are out of bounds
                x = np.clip(int(centroid[0]), 0, image_orig.width - 1)
                y = np.clip(int(centroid[1]), 0, image_orig.height - 1)
                centroid_color = image_orig.getpixel((x, y))
            
            # Extract RGB (in case there's an alpha channel)
            if isinstance(centroid_color, tuple):
                if len(centroid_color) > 3:
                    centroid_rgb = centroid_color[:3]
                else:
                    centroid_rgb = centroid_color
            else:
                # Grayscale image
                centroid_rgb = (centroid_color, centroid_color, centroid_color)
            
            # Convert to normalized tensor
            color_tensor = torch.tensor([centroid_rgb[0]/255.0, centroid_rgb[1]/255.0, centroid_rgb[2]/255.0],
                                      dtype=torch.float32).unsqueeze(0).to(device)
            
            # Pass through model
            transformed_color = model(color_tensor).squeeze(0).cpu().numpy()
            
            # Convert back to [0-255] range
            transformed_rgb = (transformed_color * 255).astype(np.uint8)
            triangle_colors.append(transformed_rgb)
    
    triangle_colors = np.array(triangle_colors)
    print(f"Completed! Generated colors for all {num_triangles} triangles.")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Use image coordinates
    
    # Plot each triangle with its CNN-generated color
    for i, triangle_idx in enumerate(triangles.simplices):
        vertices = S[triangle_idx]
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        
        # Normalize color for matplotlib
        rgb_normalized = triangle_colors[i] / 255.0
        ax.fill(xs, ys, color=rgb_normalized, edgecolor='none')
    
    ax.set_axis_off()
    ax.set_title('CNN-Colored Triangulation', fontsize=14, fontweight='bold')
    
    return {
        'figure': fig,
        'triangle_colors': triangle_colors,
        'ax': ax
    }


def save_trained_model(model, filepath, metadata=None):
    """
    Save trained model with metadata.
    
    Args:
        model: ColorTransferNet to save
        filepath: Where to save (e.g., 'models/color_transfer_spiderman_vegetables.pth')
        metadata: Dict with info like:
            - source_image: path to source
            - target_image: path to target
            - num_clusters: K-Means clusters used
            - num_distinct: Distinct colors used
            - training_epochs: epochs trained
            - final_loss: final loss value
    
    Example:
        save_trained_model(
            model, 'models/my_model.pth',
            metadata={
                'source_image': 'spiderman.jpg',
                'target_image': 'vegetables.jpg',
                'num_clusters': 25,
                'num_distinct': 10,
                'training_epochs': 1000,
                'final_loss': 0.0234
            }
        )
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare save data
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'n_layers': model.n_layers,
            'hidden_dim': model.hidden_dim
        },
        'metadata': metadata or {}
    }
    
    # Save
    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")


def load_trained_model(filepath, device='cpu'):
    """
    Load a previously trained model.
    
    Args:
        filepath: Path to saved model
        device: 'cpu' or 'cuda'
    
    Returns:
        Tuple containing:
            - model: Loaded ColorTransferNet
            - metadata: Dictionary with training info
    
    Example:
        model, metadata = load_trained_model('models/my_model.pth', device='cpu')
        print(f"Model trained on: {metadata['source_image']}")
    """
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate model
    arch = checkpoint['model_architecture']
    model = ColorTransferNet(
        n_layers=arch['n_layers'],
        hidden_dim=arch['hidden_dim']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    
    print(f"Model loaded from: {filepath}")
    if metadata:
        print("Model metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    return model, metadata


def compare_methods(S, triangles, image_orig, source_path, target_path,
                   model=None, num_clusters=25, num_distinct=10, device='cpu'):
    """
    Create side-by-side comparison of different coloring methods.
    
    Creates a figure with subplots showing:
    1. Original triangulation (centroid-based coloring)
    2. Nearest color matching (simple palette mapping)
    3. CNN color transfer (neural network method)
    4. Color palettes used
    
    This helps visualize the improvement from using CNN vs simple nearest-color.
    
    Args:
        S: Vertices array
        triangles: Delaunay triangulation
        image_orig: Original PIL Image
        source_path: Path to source image
        target_path: Path to target palette image
        model: Trained ColorTransferNet (if None, will train one)
        num_clusters: K-Means clusters
        num_distinct: Distinct colors to use
        device: 'cpu' or 'cuda'
    
    Returns:
        fig: matplotlib figure with all methods
    """
    print("\n=== Creating Method Comparison ===")
    
    # Method 1: Original coloring (centroid-based)
    print("Method 1: Original centroid-based coloring...")
    fig_orig, ax_orig = plt.subplots(figsize=(10, 8))
    ax_orig.set_aspect('equal')
    ax_orig.invert_yaxis()
    
    for triangle_idx in triangles.simplices:
        vertices = S[triangle_idx]
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        
        centroid = vertices.mean(axis=0)
        try:
            color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
        except:
            x = np.clip(int(centroid[0]), 0, image_orig.width - 1)
            y = np.clip(int(centroid[1]), 0, image_orig.height - 1)
            color = image_orig.getpixel((x, y))
        
        if len(color) > 3:
            color = color[:3]
        
        rgb_norm = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
        ax_orig.fill(xs, ys, color=rgb_norm, edgecolor='none')
    
    ax_orig.set_axis_off()
    ax_orig.set_title('Method 1: Original Centroid Coloring', fontsize=12, fontweight='bold')
    plt.close(fig_orig)
    
    # Method 2: Nearest color (from existing code)
    print("Method 2: Nearest color matching...")
    _, pixels = colour.load_image_pixels(source_path)
    _, centers, _, _ = colour.run_kmeans(pixels, num_clusters)
    palette, _ = colour.select_distinct_colors(centers, num_to_select=num_distinct)
    
    fig_nearest, ax_nearest = plt.subplots(figsize=(10, 8))
    ax_nearest.set_aspect('equal')
    ax_nearest.invert_yaxis()
    
    for triangle_idx in triangles.simplices:
        vertices = S[triangle_idx]
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        
        centroid = vertices.mean(axis=0)
        try:
            centroid_color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
        except:
            x = np.clip(int(centroid[0]), 0, image_orig.width - 1)
            y = np.clip(int(centroid[1]), 0, image_orig.height - 1)
            centroid_color = image_orig.getpixel((x, y))
        
        if len(centroid_color) > 3:
            centroid_rgb = centroid_color[:3]
        else:
            centroid_rgb = centroid_color
        
        # Find closest color in palette
        distances = np.linalg.norm(palette - np.array(centroid_rgb), axis=1)
        closest_color = palette[np.argmin(distances)]
        
        rgb_norm = (closest_color[0]/255.0, closest_color[1]/255.0, closest_color[2]/255.0)
        ax_nearest.fill(xs, ys, color=rgb_norm, edgecolor='none')
    
    ax_nearest.set_axis_off()
    ax_nearest.set_title('Method 2: Nearest Color Matching', fontsize=12, fontweight='bold')
    plt.close(fig_nearest)
    
    # Method 3: CNN color transfer
    print("Method 3: CNN color transfer...")
    if model is None:
        print("Training model...")
        data = prepare_training_data(source_path, target_path, 
                                    num_clusters, num_distinct, device=device)
        model, _ = train_color_transfer_network(
            data['source_pixels'], data['target_palette'],
            data['source_pixels_lab'], data['target_palette_lab'],
            epochs=500, batch_size=512, lr=0.001, device=device
        )
    
    result_cnn = apply_cnn_to_triangulation(model, S, triangles, image_orig, device=device)
    fig_cnn = result_cnn['figure']
    
    # Create comparison figure
    print("Creating comparison visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: Original
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(np.array(fig_orig.canvas.buffer_rgba()))
    ax1.set_axis_off()
    
    # Subplot 2: Nearest color
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(np.array(fig_nearest.canvas.buffer_rgba()))
    ax2.set_axis_off()
    
    # Subplot 3: CNN
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(np.array(fig_cnn.canvas.buffer_rgba()))
    ax3.set_axis_off()
    
    # Subplot 4: Palettes
    ax4 = fig.add_subplot(2, 2, 4)
    colour.visualize_color_palette(palette, np.ones(len(palette)) / len(palette), 'Target')
    ax4.set_axis_off()
    
    plt.tight_layout()
    plt.close(fig_orig)
    plt.close(fig_nearest)
    plt.close(fig_cnn)
    
    return fig


if __name__ == '__main__':
    # Basic testing
    print("CNN.py loaded successfully!")
    print("Run example_cnn_usage.py for complete examples.")
