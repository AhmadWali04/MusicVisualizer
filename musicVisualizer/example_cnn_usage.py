"""
example_cnn_usage.py

Example script showing how to use the CNN color transfer system.

This script demonstrates:
1. Basic usage: triangulate an image and apply a color palette using CNN
2. Reusing a previously trained model (much faster)
3. Comparing different coloring methods
4. Visualizing palette differences
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageTriangulation
import colour
import CNN
import tensorboard_feedback
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
    1. Loads spiderman.jpg and detects edges
    2. Performs Delaunay triangulation
    3. Extracts color palette from vegetables.jpg
    4. Displays interactive LAB palette visualization
    5. Trains a CNN to map colors (1000 epochs)
    6. Applies CNN to paint each triangle
    7. Shows colored result
    8. Displays feedback form (rate frequency and placement of each color)
    9. Saves image to trainingData/ with feedback-encoded filename
    10. Fine-tunes model with all previous feedback (150 epochs)
    
    Expected output:
    - Triangulated image with colors from vegetables palette
    - Interactive feedback form for rating color usage
    - Fine-tuned model incorporating your feedback
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: CNN Color Transfer with Feedback Loop")
    print("="*70)
    
    source_img = 'originalImages/spiderman.jpg'
    target_img = 'originalImages/vegetables.jpg' if os.path.exists('originalImages/vegetables.jpg') else 'hybridTheory.jpeg'
    
    # Step 1-7: Run pipeline (triangulate, train, apply)
    print("\nRunning CNN color transfer pipeline...")
    results = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        threshold=50,
        density_reduction=60,
        num_clusters=25,
        num_distinct=10,
        train_epochs=1000,
        save_model_path='models/spiderman_vegetables.pth',
        device='cpu',
        save_output=True
    )
    
    # Extract palette for visualization and feedback
    print("\nExtracting palette for feedback...")
    palette_rgb, palette_lab, percentages = colour.get_palette_for_cnn(
        target_img, num_clusters=25, num_distinct=10
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
    print("\n📊 Starting TensorBoard visualization...")
    print("   1. Open http://localhost:6006 in your browser")
    print("   2. Review the visualizations")
    print("   3. Return here to provide ratings")
    print("="*70)

    # Collect feedback using TensorBoard system
    session_name = f'spiderman_vegetables_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
        num_clusters=25, num_distinct=10,
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
        log_dir='runs/fine_tuning'
    )

    # Save fine-tuned model
    CNN.save_trained_model(
        model_ft,
        'models/spiderman_vegetables.pth',
        metadata={
            'source_image': source_img,
            'target_image': target_img,
            'num_clusters': 25,
            'num_distinct': 10,
            'training_epochs': 1000,
            'fine_tuned_epochs': 150,
            'final_loss': loss_history_ft['total'][-1] if loss_history_ft['total'] else 0
        }
    )

    # Display completion summary
    print("\n" + "="*70)
    print("✓ TRAINING & FEEDBACK CYCLE COMPLETE")
    print("="*70)
    print("\n📊 WHAT JUST HAPPENED:")
    print("   • Initial training: 1000 epochs (CNN learned color mapping)")
    print("   • Your feedback: Frequency and placement ratings for 10 colors")
    print("   • Fine-tuning: 150 epochs (model adapted to your feedback)")
    print("   • 267,523 parameters optimized")
    print("\n💾 SAVED TO:")
    print(f"   Model: models/spiderman_vegetables.pth")
    print(f"   Feedback: feedback_data/{session_name}.json")
    print(f"   TensorBoard: runs/feedback/{session_name}")
    print("\n🎯 HOW FEEDBACK WORKS:")
    print("   • Frequency: Model learns to use colors you rated low")
    print("   • Placement: Model learns better color placement")
    print("   • Recent sessions weighted more heavily")
    print("\n⚡ NEXT STEPS:")
    print("   • Run Example 1 again to incorporate more feedback")
    print("   • Run Example 2 to apply model without retraining")
    print("   • Use Example 3 for different triangulation density")
    print("\n📊 TENSORBOARD:")
    print("   • View training progress: tensorboard --logdir=runs")
    print("   • Compare sessions in browser at http://localhost:6006")
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
    Expected runtime: ~5 minutes (vs 15 minutes for Example 1)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Reusing Trained Model with Feedback")
    print("="*70)
    
    model_path = 'models/spiderman_vegetables.pth'
    
    if not os.path.exists(model_path):
        print(f"\n Model not found at {model_path}")
        print("\n HOW TO GET A MODEL:")
        print("   1. Run Example 1 first to train a model")
        print("   2. Or download a pre-trained model")
        print("   3. Place it in: models/spiderman_vegetables.pth")
        print("\n  Once you have a model:")
        print("   • Loading takes ~5 seconds")
        print("   • Applying takes ~2 minutes")
        print("   • Feedback: ~3 minutes")
        print("   • Total: ~5 minutes (3x faster than training!)")
        return None
    
    print("\n⚡ FAST MODE: Loading pre-trained model...")
    print(f"   Model file: {model_path}")
    
    source_img = 'originalImages/spiderman.jpg'
    target_img = 'originalImages/vegetables.jpg' if os.path.exists('originalImages/vegetables.jpg') else 'hybridTheory.jpeg'
    
    # Apply pre-trained model (no training step)
    results = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        use_pretrained_model=model_path,
        threshold=50,
        density_reduction=60,
        device='cpu',
        save_output=True
    )
    
    # Extract palette for visualization and feedback
    palette_rgb, palette_lab, percentages = colour.get_palette_for_cnn(
        target_img, num_clusters=25, num_distinct=10
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
    print("\n📊 Review the TensorBoard dashboard, then provide ratings")
    print("="*70)

    session_name = f'spiderman_vegetables_reuse_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    scores = tensorboard_feedback.get_user_feedback_tensorboard(
        palette_rgb, palette_lab, triangle_colors,
        image_original, image_cnn,
        session_name=session_name
    )

    # Fine-tune model with feedback
    print("\nFine-tuning model with feedback from all previous sessions...")

    data = CNN.prepare_training_data(
        source_img, target_img,
        num_clusters=25, num_distinct=10,
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
        log_dir='runs/fine_tuning'
    )

    # Save fine-tuned model
    CNN.save_trained_model(
        model_ft,
        model_path,
        metadata={
            'source_image': source_img,
            'target_image': target_img,
            'num_clusters': 25,
            'num_distinct': 10,
            'fine_tuned_epochs': 150,
            'final_loss': loss_history_ft['total'][-1] if loss_history_ft['total'] else 0
        }
    )

    # Show what just happened
    print("\n" + "="*70)
    print("✓ MODEL APPLIED & FINE-TUNED SUCCESSFULLY")
    print("="*70)
    print("\n📊 WHAT JUST HAPPENED:")
    print("   • Loaded 267,523 pre-trained parameters instantly")
    print("   • Applied model to create colored triangulation")
    print("   • Collected your feedback on color usage")
    print("   • Fine-tuned model for 150 epochs")
    print("\n⏱️  TIME COMPARISON:")
    print("   • Example 1 (train + feedback): ~18 minutes")
    print("   • Example 2 (reuse + feedback): ~5 minutes")
    print("   • Speed improvement: 3.6x faster!")
    print("\n🎯 HOW THIS WORKS:")
    print("   • Model remembered how to map colors from Example 1")
    print("   • Your feedback guided 150-epoch fine-tuning")
    print("   • Model now incorporates your preferences!")
    print("\n💡 NEXT TIME:")
    print("   • Run Example 1 or 2 again for more feedback cycles")
    print("   • Model gets smarter with each feedback loop")
    print("\n📊 TENSORBOARD:")
    print("   • View all sessions: tensorboard --logdir=runs")
    print("   • Browser: http://localhost:6006")
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
    
    model_path = 'models/spiderman_vegetables.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found. Run example_1_basic_usage() first.")
        return None
    
    print("\nTrying with fewer triangles (coarser result)...")
    results_coarse = imageTriangulation.pipeline_with_cnn(
        source_image_path='originalImages/spiderman.jpg',
        target_image_path='originalImages/vegetables.jpg' if os.path.exists('originalImages/vegetables.jpg') else 'hybridTheory.jpeg',
        use_pretrained_model=model_path,
        threshold=50,
        density_reduction=120,  # Fewer triangles
        device='cpu',
        save_output=True
    )
    
    print("\nTrying with more triangles (finer result)...")
    results_fine = imageTriangulation.pipeline_with_cnn(
        source_image_path='originalImages/spiderman.jpg',
        target_image_path='originalImages/vegetables.jpg' if os.path.exists('originalImages/vegetables.jpg') else 'hybridTheory.jpeg',
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
    Preview source and target palettes before training.
    
    This is useful for:
    - Seeing what colors will be available
    - Choosing good source/target image pairs
    - Understanding the color mapping challenge
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Palette Preview and Comparison")
    print("="*70)
    
    # Extract palettes
    print("\nExtracting source palette from: spiderman.jpg")
    source_palette_rgb, source_palette_lab, source_pct = colour.get_palette_for_cnn(
        'originalImages/spiderman.jpg', num_clusters=25, num_distinct=10
    )
    
    target_image = 'originalImages/vegetables.jpg' if os.path.exists('originalImages/vegetables.jpg') else 'hybridTheory.jpeg'
    print(f"\nExtracting target palette from: {target_image}")
    target_palette_rgb, target_palette_lab, target_pct = colour.get_palette_for_cnn(
        target_image, num_clusters=25, num_distinct=10
    )
    
    # Visualize comparison
    print("\nDisplaying palette comparison...")
    colour.visualize_palette_comparison(
        source_palette_rgb, target_palette_rgb,
        label1="Source Palette (Spiderman)",
        label2=f"Target Palette ({target_image})"
    )
    
    return {
        'source': source_palette_rgb,
        'target': target_palette_rgb
    }


def example_5_compare_methods():
    """
    Create side-by-side comparison of different coloring methods.
    
    Compares:
    1. Original centroid-based coloring
    2. Nearest color matching (simple palette mapping)
    3. CNN color transfer (neural network method)
    4. Color palettes used
    
    This demonstrates the improvements of the CNN approach.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Method Comparison")
    print("="*70)
    
    # First, load and triangulate image
    print("\nLoading and triangulating image...")
    imageTriangulation.setup_matplotlib()
    image_orig, image = imageTriangulation.load_image('originalImages/spiderman.jpg')
    image = imageTriangulation.convert_to_greyscale(image)
    image = imageTriangulation.sharpen_image(image)
    image = imageTriangulation.detect_edges(image)
    S = imageTriangulation.determine_vertices(image, threshold=50, density_reduction=60)
    triangles = imageTriangulation.Delaunay(S)
    
    # Create comparison
    print("\nCreating comparison visualization...")
    target_image = 'originalImages/vegetables.jpg' if os.path.exists('originalImages/vegetables.jpg') else 'hybridTheory.jpeg'
    
    comparison_fig = CNN.compare_methods(
        S, triangles, image_orig,
        source_path='originalImages/spiderman.jpg',
        target_path=target_image,
        model=None,  # Will train a new model
        num_clusters=25,
        num_distinct=10,
        device='cpu'
    )
    
    comparison_fig.show()
    return comparison_fig


def interactive_menu():
    """
    Interactive menu for running examples.
    """
    while True:
        print("\n" + "="*70)
        print("CNN COLOR TRANSFER EXAMPLES")
        print("="*70)
        print("\nChoose an example to run:")
        print("1. Basic usage (train and apply CNN)")
        print("2. Reuse trained model (fast)")
        print("3. Different triangulation parameters")
        print("4. Preview palettes")
        print("5. Compare coloring methods")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
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
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    # You can uncomment individual examples or use the interactive menu
    
    print("\n" + "="*70)
    print("WELCOME TO CNN COLOR TRANSFER SYSTEM")
    print("="*70)
    print("\nThis system uses neural networks to intelligently map colors")
    print("from a source image to a target palette when triangulating.")
    print("\nExamples available:")
    print("  - Basic CNN training and application")
    print("  - Reusing pre-trained models")
    print("  - Comparing different methods")
    print("  - Palette visualization")
    
    # Uncomment to run specific example:
    # example_1_basic_usage()
    
    # Or use interactive menu:
    interactive_menu()
