# Imports
import math
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
from PIL import Image
import random
from scipy.spatial import Delaunay
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import colour


def setup_matplotlib():
    """Configure matplotlib display settings."""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fontsize = 18
    params = {
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': 12,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'lines.linewidth': 1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'font.family': 'Serif',
        'font.serif': 'Hoefler Text',
        'axes.grid': False,
        'figure.figsize': (6.75, 4),
        'figure.dpi': 250,
        'mathtext.fontset': 'cm'
    }
    for param in params.keys():
        matplotlib.rcParams[param] = params[param]


def load_image(filepath):
    """Load and display original image."""
    image_orig = Image.open(filepath)
    image = Image.open(filepath)
    image.show()
    return image_orig, image


def convert_to_greyscale(image):
    """Convert image to greyscale and display."""
    image_data = image.load()
    for i in range(image.width):
        for j in range(image.height):
            try:
                r, g, b, a = image.getpixel((i, j))
            except:
                r, g, b = image.getpixel((i, j))
            grey = 0.299*r + 0.587*g + 0.114*b
            image_data[i, j] = (int(grey), int(grey), int(grey))
    image.show()
    return image


def sharpen_image(image):
    """Apply sharpening operator to image."""
    H = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    image_data = image.load()
    
    G = [[0]*image.height for i in range(image.width)]
    maxG = 0
    
    # Apply sharpening operator
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            x1 = H[0][0]*image.getpixel((i - 1, j - 1))[0]
            x2 = H[0][1]*image.getpixel((i, j - 1))[0]
            x3 = H[0][2]*image.getpixel((i + 1, j - 1))[0]
            x4 = H[1][0]*image.getpixel((i - 1, j))[0]
            x5 = H[1][1]*image.getpixel((i, j))[0]
            x6 = H[1][2]*image.getpixel((i + 1, j))[0]
            x7 = H[2][0]*image.getpixel((i - 1, j + 1))[0]
            x8 = H[2][1]*image.getpixel((i, j + 1))[0]
            x9 = H[2][2]*image.getpixel((i + 1, j + 1))[0]
            G_val = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
            G[i][j] = G_val
            
            if(G_val > maxG):
                maxG = G_val
    
    # Replace pixel data
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            image_data[i, j] = (round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255))
    image.show()
    return image


def detect_edges(image):
    """Apply Sobel edge detection operator."""
    xKernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    yKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    
    image_data = image.load()
    Gx = [[0]*image.height for i in range(image.width)]
    Gy = [[0]*image.height for i in range(image.width)]
    G = [[0]*image.height for i in range(image.width)]
    theta = [[0]*image.height for i in range(image.width)]
    
    maxG = 0
    # Apply Sobel operator
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            x1 = xKernel[0][0]*image.getpixel((i - 1, j - 1))[0]
            x2 = xKernel[0][1]*image.getpixel((i, j - 1))[0]
            x3 = xKernel[0][2]*image.getpixel((i + 1, j - 1))[0]
            x4 = xKernel[1][0]*image.getpixel((i - 1, j))[0]
            x5 = xKernel[1][1]*image.getpixel((i, j))[0]
            x6 = xKernel[1][2]*image.getpixel((i + 1, j))[0]
            x7 = xKernel[2][0]*image.getpixel((i - 1, j + 1))[0]
            x8 = xKernel[2][1]*image.getpixel((i, j + 1))[0]
            x9 = xKernel[2][2]*image.getpixel((i + 1, j + 1))[0]
            Gx_val = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
            Gx[i][j] = Gx_val
            
            y1 = yKernel[0][0]*image.getpixel((i - 1, j - 1))[0]
            y2 = yKernel[0][1]*image.getpixel((i, j - 1))[0]
            y3 = yKernel[0][2]*image.getpixel((i + 1, j - 1))[0]
            y4 = yKernel[1][0]*image.getpixel((i - 1, j))[0]
            y5 = yKernel[1][1]*image.getpixel((i, j))[0]
            y6 = yKernel[1][2]*image.getpixel((i + 1, j))[0]
            y7 = yKernel[2][0]*image.getpixel((i - 1, j + 1))[0]
            y8 = yKernel[2][1]*image.getpixel((i, j + 1))[0]
            y9 = yKernel[2][2]*image.getpixel((i + 1, j + 1))[0]
            Gy_val = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9
            Gy[i][j] = Gy_val
            
            G_val = math.sqrt(Gx_val**2 + Gy_val**2)
            G[i][j] = G_val
            
            try:
                theta[i][j] = math.atan(Gy_val/Gx_val)
            except:
                theta[i][j] = math.inf
            if(G_val > maxG):
                maxG = G_val
    
    # Replace pixel data
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            image_data[i, j] = (round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255))
    image.show()
    return image


def determine_vertices(image, threshold, density_reduction):
    """Extract vertices from edge-detected image."""
    image_data = image.load()
    S = []
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            if image_data[i, j][0] > threshold:
                S.append([i, j])
    S = random.sample(S, round(len(S)/density_reduction))  # reduce density of point cloud
    S.append([0, 0])
    S.append([0, image.height - 1])
    S.append([image.width - 1, 0])
    S.append([image.width, image.height])
    S = np.array(S)
    return S


def visualize_triangulation(S, triangles):
    """Display triangulated mesh."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.triplot(S[:,0], S[:,1], triangles.simplices, color='black')
    ax.set_axis_off()
    plt.show()


def colorize_triangulation(S, triangles, image_orig, save=False, image_name=None):
    """Color triangles based on original image and optionally save."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis()  # use image coordinates
    for triangle in range(len(triangles.simplices)):
        vertices = S[triangles.simplices[triangle]]
        a = vertices[0]
        b = vertices[1]
        c = vertices[2]
        
        xs = vertices[:,0]
        ys = vertices[:,1]
        
        centroid = [(a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3]
        color = image_orig.getpixel((centroid[0], centroid[1]))
        R = color[0]/255
        G = color[1]/255
        B = color[2]/255
        
        ax.fill(xs, ys, color=(R, G, B))
    ax.set_axis_off()
    
    if save and image_name:
        plt.savefig(os.getcwd() + '/triangulatedImages/' + image_name)
    plt.show()


def find_closest_color(target_rgb, distinct_colors):
    """
    Find the closest color from distinct_colors to target_rgb using Euclidean distance.
    
    Args:
        target_rgb: RGB tuple (r, g, b) - target color
        distinct_colors: numpy array of shape (n, 3) - distinct RGB colors
    
    Returns:
        closest_color: RGB tuple of the closest distinct color
    """
    target = np.array(target_rgb)
    distances = np.linalg.norm(distinct_colors - target, axis=1)
    closest_idx = np.argmin(distances)
    return distinct_colors[closest_idx]


def custom_color_triangulation(S, triangles, image_orig, filepath, num_clusters=10, num_distinct=10, save=False, image_name=None):
    """
    Color triangles using the closest distinct color from K-Means clustering.
    
    This function:
    1. Extracts 10 distinct colors from the image using colour.py
    2. For each triangle, gets the centroid-based color from the original image
    3. Maps that color to the closest of the 10 distinct colors
    4. Paints the triangle with only that distinct color
    
    Args:
        S: Vertices array
        triangles: Delaunay triangulation object
        image_orig: Original PIL Image
        filepath: Path to the image file
        num_clusters: Number of K-Means clusters to extract
        num_distinct: Number of distinct colors to select
        save: Whether to save the image
        image_name: Name to save the image with
    """
    # Get distinct colors from the image using colour.py functions
    _, pixels = colour.load_image_pixels(filepath)
    _, centers, _, percentages = colour.run_kmeans(pixels, num_clusters)
    distinct_centers, selected_indices = colour.select_distinct_colors(centers, num_to_select=num_distinct)
    
    print(f"\nUsing {num_distinct} distinct colors for custom triangulation:")
    for i, center in enumerate(distinct_centers):
        print(f"  Color {i + 1}: RGB{tuple(center)}")
    
    # Create the plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis()  # use image coordinates
    
    # Color each triangle with the closest distinct color
    for triangle in range(len(triangles.simplices)):
        vertices = S[triangles.simplices[triangle]]
        a = vertices[0]
        b = vertices[1]
        c = vertices[2]
        
        xs = vertices[:,0]
        ys = vertices[:,1]
        
        # Calculate centroid
        centroid = [(a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3]
        
        # Get the centroid-based color from the original image (same as colorize_triangulation does)
        centroid_color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
        
        # Extract just RGB values (in case there's an alpha channel)
        if len(centroid_color) > 3:
            centroid_rgb = centroid_color[:3]
        else:
            centroid_rgb = centroid_color
        
        # Map this centroid color to the closest distinct color
        closest_color = find_closest_color(centroid_rgb, distinct_centers)
        
        # Normalize to 0-1 range for matplotlib
        R = closest_color[0] / 255
        G = closest_color[1] / 255
        B = closest_color[2] / 255
        
        ax.fill(xs, ys, color=(R, G, B))
    
    ax.set_axis_off()
    ax.set_title(f'Custom Triangulation with {num_distinct} Distinct Colors', fontsize=12, fontweight='bold')
    
    if save and image_name:
        save_path = os.path.join(os.getcwd(), 'triangulatedImages', image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"\nSaved custom triangulation to: {save_path}")
    
    plt.show()



def main():
    """Main execution function."""
    # Command line options
    parser = OptionParser()
    parser.add_option('-d', '--d', dest='d',
                      action='store', type='int', default=60,
                      help='Density parameter')
    parser.add_option('-f', '--file', dest='filepath', default=os.getcwd() + '/MusicVisualizer/originalImages/waterLily.jpeg',
                      action='store', help='Image path for image to triangulate')
    parser.add_option('-g', '--g', dest='finalname', default='triangulated.jpeg',
                      action='store', help='Final image name for saving')
    parser.add_option('-s', '--s', dest='save',
                      action='store_true', help='Save final image')
    parser.add_option('-t', '--t', dest='t',
                      action='store', type='int', default=50,
                      help='Threshold value')
    
    (options, args) = parser.parse_args()
    
    # Setup
    setup_matplotlib()
    
    # Image processing pipeline
    image_orig, image = load_image(options.filepath)
    image = convert_to_greyscale(image)
    image = sharpen_image(image)
    image = detect_edges(image)
    
    # Vertex determination
    S = determine_vertices(image, options.t, options.d)
    
    # Delaunay triangulation
    triangles = Delaunay(S)
    
    # Visualization
    visualize_triangulation(S, triangles)
    colorize_triangulation(S, triangles, image_orig, save=options.save, image_name=options.finalname)
    
    # Custom color triangulation using distinct colors
    custom_image_name = options.finalname.replace('.jpeg', '_custom.jpeg') if options.save else None
    custom_color_triangulation(S, triangles, image_orig, options.filepath, 
                               num_clusters=10, num_distinct=10, 
                               save=options.save, image_name=custom_image_name)


if __name__ == '__main__':
    main()