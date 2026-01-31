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
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    return image_rgb, pixels


def plot_rgb_cloud(pixels, title):
    normalized = pixels / 255.0
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        pixels[:, 0],
        pixels[:, 1],
        pixels[:, 2],
        c=normalized,
        s=1,
        alpha=0.4,
        linewidth=0
    )
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_rgb_cloud_interactive(pixels, title):
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    normalized = pixels / 255.0
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pixels[:, 0],
                y=pixels[:, 1],
                z=pixels[:, 2],
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
            xaxis_title='R',
            yaxis_title='G',
            zaxis_title='B'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def run_kmeans(pixels, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    return kmeans, centers, labels, percentages


def plot_cluster_centers_3d(centers, percentages):
    normalized = centers / 255.0
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=normalized,
        s=400,
        edgecolors='black'
    )

    for i, (center, percentage) in enumerate(zip(centers, percentages)):
        label = f"RGB({center[0]}, {center[1]}, {center[2]})\n{percentage:.1f}%"
        ax.text(center[0], center[1], center[2], label, fontsize=9)

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('K-Means Cluster Centers in RGB Space')
    plt.tight_layout()
    plt.show()


def plot_cluster_centers_3d_interactive(centers, percentages):
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


def visualize_color_palette(centers, percentages):
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
    path = "hybridTheory.jpeg"
    num_clusters = 5
    use_interactive = True

    _, pixels = load_image_pixels(path)
    title = 'All Image Pixels in HybridTheory Album Cover Space'
    if use_interactive and go is not None:
        plot_rgb_cloud_interactive(pixels, title)
    else:
        plot_rgb_cloud(pixels, title)

    _, centers, _, percentages = run_kmeans(pixels, num_clusters)
    print("Most prominent RGB values:")
    for i, (center, percentage) in enumerate(zip(centers, percentages)):
        print(f"Color {i + 1}: RGB{tuple(center)} - {percentage:.1f}%")

    if use_interactive and go is not None:
        plot_cluster_centers_3d_interactive(centers, percentages)
    else:
        plot_cluster_centers_3d(centers, percentages)
    visualize_color_palette(centers, percentages)


if __name__ == '__main__':
    main()