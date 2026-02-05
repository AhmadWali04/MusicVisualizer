"""
example_cnn_usage.py

Example script showing how to use the CNN color transfer system.

This script demonstrates:
1. Basic usage: triangulate an image and apply a color palette using CNN
2. Reusing a previously trained model (much faster)
3. Comparing different coloring methods
4. Visualizing palette differences with dual popups
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # Load configuration from .env
import imageTriangulation
import colour
import CNN
import tensorboard_feedback
from plotting import visualize_dual_palettes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'bitstream vera sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
from datetime import datetime


def example_1_basic_usage():
    """
    Basic usage with FEEDBACK LOOP: triangulate, train, apply, rate, and fine-tune.

    Complete workflow:
    1. Loads template image and detects edges
    2. Performs Delaunay triangulation
    3. Extracts color palette from palette image
    4. Displays interactive LAB palette visualization
    5. Trains a CNN to map colors (1000 epochs)
    6. Applies CNN to paint each triangle
    7. Shows colored result
    8. Displays feedback form (rate frequency and placement of each color)
    9. Saves feedback to feedback_data/
    10. Fine-tunes model with all previous feedback (150 epochs)

    Expected output:
    - Triangulated image with colors from palette
    - Interactive feedback form for rating color usage
    - Fine-tuned model incorporating your feedback
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: CNN Color Transfer with Feedback Loop")
    print("="*70)

    # Print current configuration
    config.print_config()

    # Validate configuration
    errors = config.validate_config()
    if errors:
        print("\nConfiguration errors:")
        for err in errors:
            print(f"  - {err}")
        return None

    source_img = config.TEMPLATE_IMAGE
    target_img = config.PALETTE_IMAGE

    # Step 1-7: Run pipeline (triangulate, train, apply)
    print("\nRunning CNN color transfer pipeline...")
    results = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        threshold=50,
        density_reduction=config.DENSITY_REDUCTION,
        num_clusters=config.NUM_CLUSTERS,
        num_distinct=config.NUM_DISTINCT,
        train_epochs=1000,
        save_model_path=config.get_model_path(),
        device='cpu',
        save_output=True
    )

    # Extract palette for visualization and feedback
    print("\nExtracting palette for feedback...")
    palette_rgb, palette_lab, percentages = colour.get_palette_for_cnn(
        target_img, num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT
    )

    # Extract triangle colors for TensorBoard visualization
    triangle_colors = results['cnn_result']['triangle_colors']  # numpy array (N, 3)

    # Convert matplotlib figure to PIL Image for TensorBoard
    fig = results['cnn_result']['figure']
    fig.canvas.draw()

    # Use np.asarray() to automatically handle reshaping
    image_data = np.asarray(fig.canvas.buffer_rgba())
    image_data = image_data[:, :, :3]  # Remove alpha channel, keep RGB
    image_cnn = Image.fromarray(image_data)

    # Load original image for comparison
    image_original = Image.open(source_img)

    # Step 8: Collect feedback via TensorBoard
    print("\n" + "="*70)
    print("TENSORBOARD FEEDBACK COLLECTION")
    print("="*70)
    print("\n Starting TensorBoard visualization...")
    print("   1. Open http://127.0.0.1:6006 in your browser")
    print("   2. Review the visualizations")
    print("   3. Return here to provide ratings")
    print("="*70)

    # Collect feedback using TensorBoard system
    session_name = config.get_session_name(datetime.now().strftime("%Y%m%d_%H%M%S"))
    scores = tensorboard_feedback.get_user_feedback_tensorboard(
        palette_rgb, palette_lab, triangle_colors,
        image_original, image_cnn,
        session_name=session_name
    )

    # Step 10: Fine-tune model with feedback
    print("\nFine-tuning model with feedback from all previous sessions...")

    # Prepare data for fine-tuning
    data = CNN.prepare_training_data(
        source_img, target_img,
        num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT,
        use_lab=True, device='cpu'
    )

    # Fine-tune with feedback
    model = results['model']
    model_ft, loss_history_ft = CNN.fine_tune_with_feedback(
        model,
        data['source_pixels'], data['target_palette'],
        data['source_pixels_lab'], data['target_palette_lab'],
        epochs=150, batch_size=512, lr=0.0005,
        device='cpu',
        feedback_dir='feedback_data',
        log_dir='runs/fine_tuning',
        model_name=config.get_model_name()
    )

    # Save fine-tuned model
    CNN.save_trained_model(
        model_ft,
        config.get_model_path(),
        metadata={
            'source_image': source_img,
            'target_image': target_img,
            'num_clusters': config.NUM_CLUSTERS,
            'num_distinct': config.NUM_DISTINCT,
            'training_epochs': 1000,
            'fine_tuned_epochs': 150,
            'final_loss': loss_history_ft['total'][-1] if loss_history_ft['total'] else 0
        }
    )

    # Display completion summary
    print("\n" + "="*70)
    print("TRAINING & FEEDBACK CYCLE COMPLETE")
    print("="*70)
    print("\n WHAT JUST HAPPENED:")
    print("   Initial training: 1000 epochs (CNN learned color mapping)")
    print("   Your feedback: Frequency and placement ratings for 10 colors")
    print("   Fine-tuning: 150 epochs (model adapted to your feedback)")
    print("   267,523 parameters optimized")
    print("\n SAVED TO:")
    print(f"   Model: {config.get_model_path()}")
    print(f"   Feedback: feedback_data/{session_name}.json")
    print(f"   TensorBoard: runs/feedback/{session_name}")
    print("\n HOW FEEDBACK WORKS:")
    print("   Frequency: Model learns to use colors you rated low")
    print("   Placement: Model learns better color placement")
    print("   Recent sessions weighted more heavily")
    print("\n NEXT STEPS:")
    print("   Run Example 1 again to incorporate more feedback")
    print("   Run Example 2 to apply model without retraining")
    print("   Use Example 3 for different triangulation density")
    print("\n TENSORBOARD:")
    print("   View training progress: tensorboard --logdir=runs")
    print("   Compare sessions in browser at http://127.0.0.1:6006")
    print("="*70)

    return results


def example_2_reuse_trained_model():
    """
    Reuse a trained model with FEEDBACK LOOP (faster than Example 1).

    This example demonstrates:
    1. Loading a pre-trained model from disk
    2. Applying it to triangulation without retraining
    3. Collecting feedback on color usage
    4. Fine-tuning model with feedback (150 epochs)

    This is much faster than Example 1 because we skip the initial 1000-epoch training.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Reusing Trained Model with Feedback")
    print("="*70)

    # Print current configuration
    config.print_config()

    model_path = config.get_model_path()

    if not os.path.exists(model_path):
        print(f"\n Model not found at {model_path}")
        print("\n HOW TO GET A MODEL:")
        print("   1. Run Example 1 first to train a model")
        print("   2. Or download a pre-trained model")
        print(f"   3. Place it in: {model_path}")
        print("\n  Once you have a model:")
        print("   Loading takes ~5 seconds")
        print("   Applying takes ~2 minutes")
        print("   Feedback: ~3 minutes")
        print("   Total: ~5 minutes (3x faster than training!)")
        return None

    print("\n FAST MODE: Loading pre-trained model...")
    print(f"   Model file: {model_path}")

    source_img = config.TEMPLATE_IMAGE
    target_img = config.PALETTE_IMAGE

    # Apply pre-trained model (no training step)
    results = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        use_pretrained_model=model_path,
        threshold=50,
        density_reduction=config.DENSITY_REDUCTION,
        device='cpu',
        save_output=True
    )

    # Extract palette for visualization and feedback
    palette_rgb, palette_lab, percentages = colour.get_palette_for_cnn(
        target_img, num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT
    )

    # Extract triangle colors for TensorBoard
    triangle_colors = results['cnn_result']['triangle_colors']

    # Convert matplotlib figure to PIL Image
    fig = results['cnn_result']['figure']
    fig.canvas.draw()

    # Use np.asarray() to automatically handle reshaping
    image_data = np.asarray(fig.canvas.buffer_rgba())
    image_data = image_data[:, :, :3]  # Remove alpha channel, keep RGB
    image_cnn = Image.fromarray(image_data)

    # Load original image
    image_original = Image.open(source_img)

    # Collect feedback via TensorBoard
    print("\n" + "="*70)
    print("TENSORBOARD FEEDBACK COLLECTION")
    print("="*70)
    print("\n Review the TensorBoard dashboard, then provide ratings")
    print("="*70)

    session_name = config.get_session_name(f"reuse_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    scores = tensorboard_feedback.get_user_feedback_tensorboard(
        palette_rgb, palette_lab, triangle_colors,
        image_original, image_cnn,
        session_name=session_name
    )

    # Fine-tune model with feedback
    print("\nFine-tuning model with feedback from all previous sessions...")

    data = CNN.prepare_training_data(
        source_img, target_img,
        num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT,
        use_lab=True, device='cpu'
    )

    model = results['model']
    model_ft, loss_history_ft = CNN.fine_tune_with_feedback(
        model,
        data['source_pixels'], data['target_palette'],
        data['source_pixels_lab'], data['target_palette_lab'],
        epochs=150, batch_size=512, lr=0.0005,
        device='cpu',
        feedback_dir='feedback_data',
        log_dir='runs/fine_tuning',
        model_name=config.get_model_name()
    )

    # Save fine-tuned model
    CNN.save_trained_model(
        model_ft,
        model_path,
        metadata={
            'source_image': source_img,
            'target_image': target_img,
            'num_clusters': config.NUM_CLUSTERS,
            'num_distinct': config.NUM_DISTINCT,
            'fine_tuned_epochs': 150,
            'final_loss': loss_history_ft['total'][-1] if loss_history_ft['total'] else 0
        }
    )

    # Show what just happened
    print("\n" + "="*70)
    print("MODEL APPLIED & FINE-TUNED SUCCESSFULLY")
    print("="*70)
    print("\n WHAT JUST HAPPENED:")
    print("   Loaded 267,523 pre-trained parameters instantly")
    print("   Applied model to create colored triangulation")
    print("   Collected your feedback on color usage")
    print("   Fine-tuned model for 150 epochs")
    print("\n TIME COMPARISON:")
    print("   Example 1 (train + feedback): ~18 minutes")
    print("   Example 2 (reuse + feedback): ~5 minutes")
    print("   Speed improvement: 3.6x faster!")
    print("\n HOW THIS WORKS:")
    print("   Model remembered how to map colors from Example 1")
    print("   Your feedback guided 150-epoch fine-tuning")
    print("   Model now incorporates your preferences!")
    print("\n NEXT TIME:")
    print("   Run Example 1 or 2 again for more feedback cycles")
    print("   Model gets smarter with each feedback loop")
    print("\n TENSORBOARD:")
    print("   View all sessions: tensorboard --logdir=runs")
    print("   Browser: http://127.0.0.1:6006")
    print("="*70)

    return results


def example_3_different_triangulation_params():
    """
    Use different triangulation parameters with same trained model.

    This example shows how the triangulation density affects the final result:
    - Higher density_reduction = fewer, larger triangles = coarser result
    - Lower density_reduction = more, smaller triangles = finer detail

    You can reuse the same trained model with different geometric parameters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Different Triangulation Parameters")
    print("="*70)

    # Print current configuration
    config.print_config()

    model_path = config.get_model_path()

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run example_1_basic_usage() first.")
        return None

    source_img = config.TEMPLATE_IMAGE
    target_img = config.PALETTE_IMAGE

    print("\nTrying with fewer triangles (coarser result)...")
    results_coarse = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        use_pretrained_model=model_path,
        threshold=50,
        density_reduction=120,  # Fewer triangles
        device='cpu',
        save_output=True
    )

    print("\nTrying with more triangles (finer result)...")
    results_fine = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        use_pretrained_model=model_path,
        threshold=50,
        density_reduction=30,  # More triangles
        device='cpu',
        save_output=True
    )

    print("\nComparison complete! Notice how triangulation density affects detail.")
    return {'coarse': results_coarse, 'fine': results_fine}


def example_4_palette_preview():
    """
    Preview palette colors with DUAL POPUP windows.

    This shows:
    - Popup 1: 10 colors from K-means clustering of the palette image
    - Popup 2: Top 5 most distinct colors selected by LAB algorithm

    This helps understand the color selection process:
    1. K-means finds dominant color clusters
    2. LAB algorithm selects most perceptually distinct colors
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Palette Preview with Dual Popups")
    print("="*70)

    # Print current configuration
    config.print_config()

    palette_image = config.PALETTE_IMAGE
    palette_name = config.get_image_basename(palette_image)

    print(f"\nExtracting palette from: {palette_image}")

    # Step 1: Load image and convert to LAB
    _, pixels = colour.load_image_pixels(palette_image)
    pixels_lab = colour.convert_rgb_pixels_to_lab(pixels)

    # Step 2: Run K-means clustering
    print(f"Running K-Means clustering ({config.NUM_CLUSTERS} clusters)...")
    _, centers_lab, _, _ = colour.run_kmeans_lab(pixels_lab, config.NUM_CLUSTERS)

    # Convert LAB centers to RGB
    centers_rgb = colour.convert_lab_centers_to_rgb(centers_lab)

    # Step 3: Select 10 distinct colors using LAB algorithm
    print("Selecting 10 distinct colors using LAB algorithm...")
    palette_lab_10, palette_rgb_10, _ = colour.select_distinct_colors_lab(
        centers_lab, centers_rgb, num_to_select=10
    )

    # Step 4: Select top 5 most distinct from those 10
    print("Selecting top 5 most distinct from the 10...")
    palette_lab_5, palette_rgb_5, _ = colour.select_distinct_colors_lab(
        palette_lab_10, palette_rgb_10, num_to_select=5
    )

    # Print the colors
    print("\n" + "-"*50)
    print("10 Clustered Colors:")
    for i, rgb in enumerate(palette_rgb_10):
        print(f"  Color {i+1}: RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})")

    print("\n5 LAB-Selected Distinct Colors:")
    for i, rgb in enumerate(palette_rgb_5):
        print(f"  Color {i+1}: RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})")
    print("-"*50)

    # Step 5: Display dual popups
    print("\nDisplaying dual palette visualization...")
    print("  - Popup 1: All 10 clustered colors")
    print("  - Popup 2: Top 5 most distinct (LAB selection)")

    visualize_dual_palettes(
        palette_rgb_10,
        palette_rgb_5,
        title1=f"10 Clustered Colors from {palette_name}",
        title2="Top 5 Most Distinct (LAB Algorithm)"
    )

    return {
        'all_10_colors': palette_rgb_10,
        'top_5_distinct': palette_rgb_5,
        'palette_lab_10': palette_lab_10,
        'palette_lab_5': palette_lab_5
    }


def example_5_compare_methods():
    """
    Visualize color palette and RGB distribution.

    This example shows:
    1. Basic triangulation with original colors
    2. Dominant color palette extraction
    3. Interactive RGB color cloud visualization

    No CNN training - this is a quick visualization tool.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Color Palette Visualization")
    print("="*70)

    # Print current configuration
    config.print_config()

    source_img = config.TEMPLATE_IMAGE
    target_img = config.PALETTE_IMAGE

    # Step 1: Load and triangulate image
    print("\nLoading and triangulating image...")
    imageTriangulation.setup_matplotlib()
    image_orig, image = imageTriangulation.load_image(source_img)
    image = imageTriangulation.convert_to_greyscale(image)
    image = imageTriangulation.sharpen_image(image)
    image = imageTriangulation.detect_edges(image)
    S = imageTriangulation.determine_vertices(
        image,
        threshold=50,
        density_reduction=config.DENSITY_REDUCTION
    )
    triangles = imageTriangulation.Delaunay(S)

    # Step 2: Extract palette from target image
    print("\nExtracting color palette...")
    palette_rgb, palette_lab, percentages = colour.get_palette_for_cnn(
        target_img,
        num_clusters=config.NUM_CLUSTERS,
        num_distinct=config.NUM_DISTINCT
    )

    # Step 3: Colorize with original image colors and save to standardColored subdirectory
    print("\nColorizing triangulation...")
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.getcwd(), 'triangulatedImages', 'standardColored', config.get_model_name())
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'standard_{timestamp}.png')

    imageTriangulation.colorize_triangulation(
        S, triangles, image_orig,
        save=True,
        image_name=output_path  # Full path including subdirectories
    )

    # Step 4: Show dominant palette
    print("\nDisplaying color palette...")
    from plotting import visualize_color_palette
    visualize_color_palette(
        palette_rgb,
        percentages,
        'Target Palette Colors'
    )

    # Step 5: Show RGB cloud
    print("\nGenerating RGB color cloud...")
    _, pixels = colour.load_image_pixels(target_img)
    from plotting import plot_rgb_cloud_interactive
    plot_rgb_cloud_interactive(
        pixels,
        f'RGB Color Space - {config.get_image_basename(target_img)}',
        max_points=50000
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nWhat you see:")
    print("  1. Triangulated image with original colors")
    print("  2. Dominant color palette from target image (bar chart)")
    print("  3. 3D RGB color cloud (interactive)")
    print(f"\nImage saved to: {output_path}")
    print("\nThis is a fast, non-ML visualization method.")
    print("Use Examples 1-2 for CNN-based intelligent color transfer.")
    print("="*70)

    return {
        'triangulation': (S, triangles),
        'palette_rgb': palette_rgb,
        'palette_lab': palette_lab,
        'percentages': percentages,
        'output_path': output_path
    }


def interactive_menu():
    """
    Interactive menu for running examples.
    """
    while True:
        print("\n" + "="*70)
        print("CNN COLOR TRANSFER EXAMPLES")
        print("="*70)

        # Show current config
        print(f"\nCurrent configuration:")
        print(f"  Template: {config.TEMPLATE_IMAGE}")
        print(f"  Palette:  {config.PALETTE_IMAGE}")
        print(f"  Model:    {config.get_model_path()}")

        print("\nChoose an example to run:")
        print("1. Basic usage (train and apply CNN)")
        print("2. Reuse trained model (fast)")
        print("3. Different triangulation parameters")
        print("4. Preview palettes (dual popup)")
        print("5. Compare coloring methods")
        print("c. Show current configuration")
        print("0. Exit")

        choice = input("\nEnter your choice (0-5, c): ").strip().lower()

        if choice == '0':
            print("Exiting.")
            break
        elif choice == '1':
            example_1_basic_usage()
        elif choice == '2':
            example_2_reuse_trained_model()
        elif choice == '3':
            example_3_different_triangulation_params()
        elif choice == '4':
            example_4_palette_preview()
        elif choice == '5':
            example_5_compare_methods()
        elif choice == 'c':
            config.print_config()
            errors = config.validate_config()
            if errors:
                print("\nConfiguration errors:")
                for err in errors:
                    print(f"  - {err}")
            else:
                print("\nConfiguration is valid!")
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    # Print welcome banner
    print("\n" + "="*70)
    print("WELCOME TO CNN COLOR TRANSFER SYSTEM")
    print("="*70)
    print("\nThis system uses neural networks to intelligently map colors")
    print("from a source image to a target palette when triangulating.")

    # Show current configuration
    print("\n" + "-"*70)
    config.print_config()
    print("-"*70)

    print("\nExamples available:")
    print("  1. Basic CNN training and application")
    print("  2. Reusing pre-trained models")
    print("  3. Different triangulation parameters")
    print("  4. Palette visualization (dual popup)")
    print("  5. Comparing different methods")

    print("\nTo change images/parameters, edit .env file or set environment variables.")

    # Use interactive menu
    interactive_menu()
