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
    
    Args:
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


def visualize_color_palette(centers, percentages):
    """
    Description:
    Visualize the final color pallete as a dominant bar graph
    --------------
    Parameters:
    centers: The RGB clusters themselves
    pecentages: The percent that those clusters composet the image of 
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
    ax.set_title('Dominant Color Palette', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()


def main():
    path = "originalImages/spiderman.jpg"
    #path = "hybridTheory.jpeg"
    

    num_clusters = 10
    num_distinct = 10
    use_interactive = True

    _, pixels = load_image_pixels(path)
    title = 'All Image Pixels in HybridTheory Album Cover Space'
    # plot the rgb cloud
    plot_rgb_cloud_interactive(pixels, title)
    
    _, centers, _, percentages = run_kmeans(pixels, num_clusters)
    print("\nAll K-Means cluster centers:")
    for i, (center, percentage) in enumerate(zip(centers, percentages)):
        print(f"Color {i + 1}: RGB{tuple(center)} - {percentage:.1f}%")
    
    # Select distinct colors using greedy max-min distance
    distinct_centers, selected_indices = select_distinct_colors(centers, num_to_select=num_distinct)
    distinct_percentages = percentages[selected_indices]
    
    print(f"\nSelected {num_distinct} distinct colors:")
    for i, (center, percentage) in enumerate(zip(distinct_centers, distinct_percentages)):
        print(f"Color {i + 1}: RGB{tuple(center)} - {percentage:.1f}%")

    plot_cluster_centers_3d_interactive(distinct_centers, distinct_percentages)
    plot_cluster_centers_lab_interactive(distinct_centers, distinct_percentages)
    visualize_color_palette(distinct_centers, distinct_percentages)


if __name__ == '__main__':
    main()