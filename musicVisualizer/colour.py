import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import plotly.graph_objects as go
except ImportError:  # Plotly is optional; matplotlib remains as fallback
    go = None


def load_image_pixels(image_path):
    """
    Description:
    In this case, we read the pixels and convert it into a 2D array of RGB values.
    --------------
    Parameters:
    image_path: Path to the image file
    --------------
    Returns:
    image_rgb: Image in RGB format
    pixels: 2D array of shape (num_pixels, 3) with RGB values
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Handle alpha channel if present
    if image_rgb.shape[-1] == 4:  # RGBA
        image_rgb = image_rgb[:, :, :3]  # Keep only RGB channels
    
    # Validate and clean data
    pixels = image_rgb.reshape(-1, 3)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    
    # Diagnostics
    print(f"Image shape: {image_rgb.shape}")
    print(f"Pixels data shape: {pixels.shape}")
    print(f"Data type: {pixels.dtype}")
    print(f"Min value: {pixels.min()}, Max value: {pixels.max()}")
    print(f"First 5 pixels:\n{pixels[:5]}")
    return image_rgb, pixels


def convert_rgb_pixels_to_lab(pixels_rgb):
    """
    Convert RGB pixel array to LAB color space.
    
    Args:
        pixels_rgb: Array of shape (N, 3) with RGB values [0-255]
    
    Returns:
        pixels_lab: Array of shape (N, 3) with LAB values
    """
    # Reshape to image format for cv2
    h = int(np.sqrt(len(pixels_rgb)))
    w = len(pixels_rgb) // h
    remainder = len(pixels_rgb) % h
    if remainder != 0:
        # Add padding if needed
        padding = h - remainder
        pixels_rgb_padded = np.vstack([pixels_rgb, np.zeros((padding, 3), dtype=np.uint8)])
        h = int(np.sqrt(len(pixels_rgb_padded)))
        w = len(pixels_rgb_padded) // h
        img_rgb = pixels_rgb_padded[:h*w].reshape(h, w, 3)
    else:
        img_rgb = pixels_rgb.reshape(h, w, 3)
    
    # Convert to LAB
    img_lab = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    pixels_lab = img_lab.reshape(-1, 3)
    
    # Remove padding if we added any
    if remainder != 0:
        pixels_lab = pixels_lab[:len(pixels_rgb)]
    
    return pixels_lab

def plot_rgb_cloud_interactive(pixels, title, max_points=50000):
    """
    Description:
    This function creates a 3D scatter plot using Plotly to visualize 
    the distribution of RGB pixel values in 3D space.
    --------------
    Parameters:
    pixels: Array of RGB pixel values (N, 3)
    title: Plot title
    max_points: Maximum number of points to plot (default: 50000)
    --------------
    Returns:
    Nothing, but it shows the cloud of all the points plotted
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")
    
    # Downsample if too many points
    if len(pixels) > max_points:
        indices = np.random.choice(len(pixels), max_points, replace=False)
        pixels_sampled = pixels[indices]
        print(f"Downsampled from {len(pixels)} to {max_points} points for visualization")
    else:
        pixels_sampled = pixels
    
    normalized = pixels_sampled / 255.0  # Normalize RGB values to [0, 1] for color mapping
    
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pixels_sampled[:, 0],
                y=pixels_sampled[:, 1],
                z=pixels_sampled[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=normalized,
                    opacity=0.5
                )
            )
        ]
    )
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='R', range=[0, 255]),
            yaxis=dict(title='G', range=[0, 255]),
            zaxis=dict(title='B', range=[0, 255])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_lab_cloud_interactive(pixels_lab, pixels_rgb, title, max_points=50000):
    """
    Plot LAB color space cloud with colors from original RGB values.
    
    Args:
        pixels_lab: Array of LAB pixel values (N, 3)
        pixels_rgb: Array of RGB pixel values (N, 3) for coloring
        title: Plot title
        max_points: Maximum number of points to plot
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")
    
    # Downsample if too many points
    if len(pixels_lab) > max_points:
        indices = np.random.choice(len(pixels_lab), max_points, replace=False)
        pixels_lab_sampled = pixels_lab[indices]
        pixels_rgb_sampled = pixels_rgb[indices]
        print(f"Downsampled from {len(pixels_lab)} to {max_points} points for LAB visualization")
    else:
        pixels_lab_sampled = pixels_lab
        pixels_rgb_sampled = pixels_rgb
    
    # Use RGB colors for the points
    colors = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in pixels_rgb_sampled]
    
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pixels_lab_sampled[:, 0],  # L
                y=pixels_lab_sampled[:, 1],  # A
                z=pixels_lab_sampled[:, 2],  # B
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors,
                    opacity=0.5
                )
            )
        ]
    )
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='L (Lightness)', range=[0, 100]),
            yaxis=dict(title='A (Green-Red)', range=[-128, 127]),
            zaxis=dict(title='B (Blue-Yellow)', range=[-128, 127])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def run_kmeans(pixels, num_clusters):
    """
    Description:
    Running K-Means to find colours
    -------------
    Parameters:
    pixels : Array of RGB pixel data
    num_clusters: a specification on the number of clusters we want
    -------------
    Returns
    kmeans : Kmeans objects after fitting
    centers: Cluster centers in RGB format (The coordinates of each cluster center)
    labels: Labels assigned to each pixel (Which cluster each pixel belongs to)
    percentages: A numpy array containing the percentage of each cluster (how much of the image it makes up)
    -------------
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    return kmeans, centers, labels, percentages


def run_kmeans_lab(pixels_lab, num_clusters=5):
    """
    Run K-Means clustering in LAB color space.
    
    Args:
        pixels_lab: Array of LAB pixel values
        num_clusters: Number of clusters
    
    Returns:
        kmeans: KMeans object
        centers_lab: Cluster centers in LAB format
        labels: Pixel cluster assignments
        percentages: Percentage of pixels in each cluster
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels_lab)

    centers_lab = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    return kmeans, centers_lab, labels, percentages


def plot_cluster_centers_lab_kmeans_interactive(centers_lab, percentages, centers_rgb):
    """
    Plot LAB K-Means cluster centers colored by their RGB equivalents.
    
    Args:
        centers_lab: Cluster centers in LAB space
        percentages: Percentage distribution
        centers_rgb: Corresponding RGB values for coloring
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    colors = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in centers_rgb]
    labels = [
        f"LAB({int(c[0])}, {int(c[1])}, {int(c[2])})<br>RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})<br>{p:.1f}%"
        for c, rgb, p in zip(centers_lab, centers_rgb, percentages)
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=centers_lab[:, 0],
                y=centers_lab[:, 1],
                z=centers_lab[:, 2],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(color='black', width=1)
                )
            )
        ]
    )
    fig.update_layout(
        title='K-Means Cluster Centers in LAB Space',
        scene=dict(
            xaxis=dict(title='L (Lightness)'),
            yaxis=dict(title='A (Green-Red)'),
            zaxis=dict(title='B (Blue-Yellow)')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_cluster_centers_3d_interactive(centers, percentages):
    """
    Description:
    Take the Kmeans data we just computed, and now we plot it
    --------------
    Parameters:
    centers: The coordinaes of each cluster object
    percentages: The percentage of the total iamge that each colour is present it
    --------------
    Returns:
    Nothing, but it displays the plot
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    normalized = centers / 255.0
    labels = [
        f"RGB({c[0]}, {c[1]}, {c[2]})<br>{p:.1f}%"
        for c, p in zip(centers, percentages)
    ]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=8,
                    color=normalized,
                    line=dict(color='black', width=1)
                )
            )
        ]
    )
    fig.update_layout(
        title='K-Means Cluster Centers in RGB Space',
        scene=dict(
            xaxis_title='R',
            yaxis_title='G',
            zaxis_title='B'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def convert_rgb_centers_to_lab(centers):
    """
    Description:
    Converts RGB scale to LAB scale
    --------------
    Parameters:
    centers: The RGB clusters
    --------------
    Returns:
    centers_lab: The RGB clusters but now in LAB
    """
    centers_rgb = centers.astype(np.uint8).reshape(1, -1, 3)
    centers_lab = cv2.cvtColor(centers_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return centers_lab

def select_distinct_colors(centers, num_to_select):
    """
    Select distinct colors using Greedy Max-Min Distance algorithm in LAB space.
    
    Args
        centers: RGB color centers from K-Means (shape: [n_clusters, 3])
        num_to_select: Number of distinct colors to select
    
    Returns:
        selected_rgb: Selected colors in RGB format
        selected_indices: Indices of selected colors in original centers array
    """
    # Convert to LAB space for perceptual distance
    centers_lab = convert_rgb_centers_to_lab(centers)
    
    num_colors = len(centers)
    if num_to_select >= num_colors:
        return centers, np.arange(num_colors)
    
    selected_indices = []
    remaining_indices = list(range(num_colors))
    
    # Step 1: Select color with highest L (lightness) value
    lightness_values = centers_lab[:, 0]
    first_idx = np.argmax(lightness_values)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Step 2: Iteratively select colors with maximum distance from nearest selected color
    while len(selected_indices) < num_to_select:
        max_min_distance = -1
        next_color_idx = None
        
        for candidate_idx in remaining_indices:
            # Calculate distance to all already-selected colors
            candidate_lab = centers_lab[candidate_idx]
            min_distance = float('inf')
            
            for selected_idx in selected_indices:
                selected_lab = centers_lab[selected_idx]
                # Euclidean distance in LAB space
                distance = np.linalg.norm(candidate_lab - selected_lab)
                min_distance = min(min_distance, distance)
            
            # Track candidate with maximum minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_color_idx = candidate_idx
        
        selected_indices.append(next_color_idx)
        remaining_indices.remove(next_color_idx)
    
    selected_rgb = centers[selected_indices]
    return selected_rgb, np.array(selected_indices)


def select_distinct_colors_lab(centers_lab, centers_rgb, num_to_select=5):
    """
    Select distinct colors using Greedy Max-Min Distance algorithm in LAB space.
    Works directly with LAB centers.
    
    Args:
        centers_lab: LAB color centers from K-Means (shape: [n_clusters, 3])
        centers_rgb: Corresponding RGB values (shape: [n_clusters, 3])
        num_to_select: Number of distinct colors to select
    
    Returns:
        selected_lab: Selected colors in LAB format
        selected_rgb: Selected colors in RGB format
        selected_indices: Indices of selected colors
    """
    num_colors = len(centers_lab)
    if num_to_select >= num_colors:
        return centers_lab, centers_rgb, np.arange(num_colors)
    
    selected_indices = []
    remaining_indices = list(range(num_colors))
    
    # Step 1: Select color with highest L (lightness) value
    lightness_values = centers_lab[:, 0]
    first_idx = np.argmax(lightness_values)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Step 2: Iteratively select colors with maximum distance from nearest selected color
    while len(selected_indices) < num_to_select:
        max_min_distance = -1
        next_color_idx = None
        
        for candidate_idx in remaining_indices:
            candidate_lab = centers_lab[candidate_idx]
            min_distance = float('inf')
            
            for selected_idx in selected_indices:
                selected_lab = centers_lab[selected_idx]
                distance = np.linalg.norm(candidate_lab - selected_lab)
                min_distance = min(min_distance, distance)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_color_idx = candidate_idx
        
        selected_indices.append(next_color_idx)
        remaining_indices.remove(next_color_idx)
    
    selected_lab = centers_lab[selected_indices]
    selected_rgb = centers_rgb[selected_indices]
    return selected_lab, selected_rgb, np.array(selected_indices)



def plot_cluster_centers_lab_interactive(centers, percentages):
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    centers_lab = convert_rgb_centers_to_lab(centers)
    normalized = centers / 255.0
    labels = [
        f"LAB({c_lab[0]}, {c_lab[1]}, {c_lab[2]})<br>RGB({c_rgb[0]}, {c_rgb[1]}, {c_rgb[2]})<br>{p:.1f}%"
        for c_lab, c_rgb, p in zip(centers_lab, centers, percentages)
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=centers_lab[:, 0],
                y=centers_lab[:, 1],
                z=centers_lab[:, 2],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=8,
                    color=normalized,
                    line=dict(color='black', width=1)
                )
            )
        ]
    )
    fig.update_layout(
        title='K-Means Cluster Centers in LAB Space',
        scene=dict(
            xaxis_title='L',
            yaxis_title='A',
            zaxis_title='B'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def visualize_color_palette(centers, percentages, color_space):
    """
    Description:
    Visualize the final color pallete as a dominant bar graph
    --------------
    Parameters:
    centers: The RGB clusters themselves
    pecentages: The percent that those clusters composet the image of
    color_space: Either 'RGB' or 'LAB' to determine the title
    --------------
    Returns:
    Nothing, but does show a graph
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    x_position = 0
    for center, percentage in zip(centers, percentages):
        normalized_color = center / 255.0
        width = percentage
        rect = Rectangle((x_position, 0), width, 1, facecolor=normalized_color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        label = f"RGB({center[0]}, {center[1]}, {center[2]})\n{percentage:.1f}%"
        ax.text(
            x_position + width / 2,
            0.5,
            label,
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            color='white' if sum(center) < 384 else 'black'
        )
        x_position += width
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    
    ax.set_title(f'Dominant {color_space} Color Palette', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()


def get_palette_for_cnn(image_path, num_clusters=25, num_distinct=10, use_lab=True):
    """
    Extract color palette optimized for CNN training.
    
    This is a convenience wrapper that combines existing functions
    to produce exactly what CNN.py needs.
    
    Args:
        image_path: Path to palette/style image
        num_clusters: Number of K-Means clusters (default: 25)
        num_distinct: Number of distinct colors to select (default: 10)
        use_lab: If True, use LAB color space (recommended - default: True)
    
    Returns:
        Tuple containing:
            - palette_rgb: numpy array (num_distinct, 3) - RGB colors [0-255]
            - palette_lab: numpy array (num_distinct, 3) - LAB colors
            - percentages: numpy array (num_distinct,) - percentage distribution
    
    Example:
        palette_rgb, palette_lab, percentages = get_palette_for_cnn(
            'vegetables.jpg', num_clusters=25, num_distinct=10
        )
        print(f"Extracted {len(palette_rgb)} distinct colors")
    """
    print(f"\nExtracting palette from: {image_path}")
    
    # Load image
    _, pixels = load_image_pixels(image_path)
    pixels_lab = convert_rgb_pixels_to_lab(pixels)
    
    if use_lab:
        # Use LAB color space for clustering
        print(f"Running K-Means in LAB space ({num_clusters} clusters)...")
        _, centers_lab, _, percentages = run_kmeans_lab(pixels_lab, num_clusters)
        
        # Convert LAB centers to RGB for output
        centers_rgb = []
        for lab in centers_lab:
            lab_img = np.array([[lab]], dtype=np.uint8)
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            centers_rgb.append(rgb_img[0, 0])
        centers_rgb = np.array(centers_rgb)
        
        # Select distinct colors
        print(f"Selecting {num_distinct} distinct colors...")
        palette_lab, palette_rgb, _ = select_distinct_colors_lab(
            centers_lab, centers_rgb, num_to_select=num_distinct
        )
        percentages_selected = percentages[_]
    else:
        # Use RGB color space for clustering
        print(f"Running K-Means in RGB space ({num_clusters} clusters)...")
        _, centers_rgb, _, percentages = run_kmeans(pixels, num_clusters)
        
        # Select distinct colors
        print(f"Selecting {num_distinct} distinct colors...")
        palette_rgb, _ = select_distinct_colors(centers_rgb, num_to_select=num_distinct)
        percentages_selected = percentages[_]
        
        # Convert palette to LAB for reference
        palette_rgb_uint8 = palette_rgb.astype(np.uint8).reshape(1, -1, 3)
        palette_lab = cv2.cvtColor(palette_rgb_uint8, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    
    print(f"Palette extracted: {len(palette_rgb)} colors")
    return palette_rgb, palette_lab, percentages_selected


def visualize_palette_comparison(palette1_rgb, palette2_rgb,
                                label1="Palette 1", label2="Palette 2"):
    """
    Visualize two color palettes side by side.
    
    Useful for comparing source vs target palettes before CNN training.
    
    Args:
        palette1_rgb: numpy array (N, 3) - first palette, RGB values [0-255]
        palette2_rgb: numpy array (M, 3) - second palette, RGB values [0-255]
        label1: Label for first palette (default: "Palette 1")
        label2: Label for second palette (default: "Palette 2")
    
    Example:
        source_palette, _, _ = get_palette_for_cnn('spiderman.jpg')
        target_palette, _, _ = get_palette_for_cnn('vegetables.jpg')
        visualize_palette_comparison(source_palette, target_palette, 
                                    "Spiderman Colors", "Vegetable Colors")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Palette 1
    x_position = 0
    for center in palette1_rgb:
        normalized_color = center / 255.0
        width = 1.0 / len(palette1_rgb)
        rect = Rectangle((x_position, 0), width, 1, 
                        facecolor=normalized_color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        x_position += width
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(label1, fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('auto')
    
    # Palette 2
    x_position = 0
    for center in palette2_rgb:
        normalized_color = center / 255.0
        width = 1.0 / len(palette2_rgb)
        rect = Rectangle((x_position, 0), width, 1,
                        facecolor=normalized_color, edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        x_position += width
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(label2, fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('auto')
    
    plt.tight_layout()
    plt.show()


def main():
    path = "originalImages/spiderman.jpg"
    #path = "hybridTheory.jpeg"
    
    num_clusters = 25
    num_distinct = 10 
    use_interactive = True

    # Load image and convert to LAB
    _, pixels = load_image_pixels(path)
    pixels_lab = convert_rgb_pixels_to_lab(pixels)
    print(f"\nLAB pixels shape: {pixels_lab.shape}")
    print(f"LAB range - L: [{pixels_lab[:, 0].min()}, {pixels_lab[:, 0].max()}], A: [{pixels_lab[:, 1].min()}, {pixels_lab[:, 1].max()}], B: [{pixels_lab[:, 2].min()}, {pixels_lab[:, 2].max()}]")
    
    # Plot RGB cloud
    print("\n=== RGB Color Cloud ===")
    title_rgb = 'All Image Pixels in RGB Color Space'
    plot_rgb_cloud_interactive(pixels, title_rgb)
    
    # Plot LAB cloud
    print("\n=== LAB Color Cloud ===")
    title_lab = 'All Image Pixels in LAB Color Space'
    plot_lab_cloud_interactive(pixels_lab, pixels, title_lab)
    
    # RGB K-Means clustering
    print("\n=== RGB K-Means Clustering ===")
    _, centers_rgb, _, percentages_rgb = run_kmeans(pixels, num_clusters)
    print("\nAll RGB K-Means cluster centers:")
    for i, (center, percentage) in enumerate(zip(centers_rgb, percentages_rgb)):
        print(f"Color {i + 1}: RGB{tuple(center)} - {percentage:.1f}%")
    
    # LAB K-Means clustering
    print("\n=== LAB K-Means Clustering ===")
    _, centers_lab, _, percentages_lab = run_kmeans_lab(pixels_lab, num_clusters)
    
    # Convert LAB centers back to RGB for display
    centers_lab_as_rgb = []
    for lab in centers_lab:
        lab_img = np.array([[lab]], dtype=np.uint8)
        rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        centers_lab_as_rgb.append(rgb_img[0, 0])
    centers_lab_as_rgb = np.array(centers_lab_as_rgb)
    
    print("\nAll LAB K-Means cluster centers:")
    for i, (center_lab, center_rgb, percentage) in enumerate(zip(centers_lab, centers_lab_as_rgb, percentages_lab)):
        print(f"Color {i + 1}: LAB({int(center_lab[0])}, {int(center_lab[1])}, {int(center_lab[2])}) = RGB{tuple(center_rgb)} - {percentage:.1f}%")
    
    # Plot RGB cluster centers
    if use_interactive and go is not None:
        plot_cluster_centers_3d_interactive(centers_rgb, percentages_rgb)
    
    # Plot LAB cluster centers
    if use_interactive and go is not None:
        plot_cluster_centers_lab_kmeans_interactive(centers_lab, percentages_lab, centers_lab_as_rgb)
    
    # Select distinct RGB colors
    print("\n=== Distinct RGB Colors ===")
    distinct_centers_rgb, selected_indices_rgb = select_distinct_colors(centers_rgb, num_to_select=num_distinct)
    distinct_percentages_rgb = percentages_rgb[selected_indices_rgb]
    
    print(f"\nSelected {num_distinct} distinct RGB colors:")
    for i, (center, percentage) in enumerate(zip(distinct_centers_rgb, distinct_percentages_rgb)):
        print(f"Color {i + 1}: RGB{tuple(center)} - {percentage:.1f}%")
    
    # Select distinct LAB colors
    print("\n=== Distinct LAB Colors ===")
    distinct_centers_lab, distinct_centers_lab_as_rgb, selected_indices_lab = select_distinct_colors_lab(
        centers_lab, centers_lab_as_rgb, num_to_select=num_distinct
    )
    distinct_percentages_lab = percentages_lab[selected_indices_lab]
    
    print(f"\nSelected {num_distinct} distinct LAB colors:")
    for i, (center_lab, center_rgb, percentage) in enumerate(zip(distinct_centers_lab, distinct_centers_lab_as_rgb, distinct_percentages_lab)):
        print(f"Color {i + 1}: LAB({int(center_lab[0])}, {int(center_lab[1])}, {int(center_lab[2])}) = RGB{tuple(center_rgb)} - {percentage:.1f}%")
    
    # Visualize distinct RGB colors palette
    print("\n=== RGB Color Palette ===")
    visualize_color_palette(distinct_centers_rgb, distinct_percentages_rgb, 'RGB')
    
    # Visualize distinct LAB colors palette
    print("\n=== LAB Color Palette ===")
    visualize_color_palette(distinct_centers_lab_as_rgb, distinct_percentages_lab, 'LAB')


if __name__ == '__main__':
    main()