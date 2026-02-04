# CNN Color Transfer - Quick Start Guide

## 30-Second Setup

```bash
# You already have everything installed!
# PyTorch was just added to your venv
cd /Users/ahmadwali04/Desktop/personal/Projects/musicVisualizer
source .venv/bin/activate
```

## Run Your First Example

### Option 1: Interactive Menu (Recommended)

```bash
python example_cnn_usage.py
# Choose option 1, 2, or 4 from the menu
```

### Option 2: Direct Python

```python
from MusicVisualizer import imageTriangulation

results = imageTriangulation.pipeline_with_cnn(
    source_image_path='originalImages/spiderman.jpg',
    target_image_path='hybridTheory.jpeg',
    train_epochs=500  # Start with 500 for testing
)

results['cnn_result']['figure'].show()
```

### Option 3: Preview Palettes First

```python
import colour

# See what colors you'll be working with
source_palette, _, _ = colour.get_palette_for_cnn('originalImages/spiderman.jpg')
target_palette, _, _ = colour.get_palette_for_cnn('hybridTheory.jpeg')

colour.visualize_palette_comparison(source_palette, target_palette)
```

## What Each Example Does

| Example | Time | Use Case |
|---------|------|----------|
| 1. Basic Usage | 10-15 min | First time, see full pipeline |
| 2. Reuse Model | 30 sec | Apply saved model (if trained) |
| 3. Triangulation Params | 10-15 min | Experiment with geometry |
| 4. Palette Preview | 10 sec | See colors before training |
| 5. Compare Methods | 10-15 min | See CNN vs other methods |

## Expected Output

After training, you'll see:

1. **Training Progress**
   ```
   Epoch 50/500 - Total: 0.234567, Histogram: ..., Nearest: ..., Smoothness: ...
   Epoch 100/500 - ...
   ```

2. **Training Visualization** (6 subplots)
   - Source color cloud
   - Target palette
   - Network output
   - Loss curves
   - Component losses
   - Color mapping examples

3. **Colored Triangulation** (Main Result)
   - Image with triangles colored using CNN

## Troubleshooting

### Problem: Training takes too long

**Solution**: Reduce epochs or use GPU

```python
# Fewer epochs = faster training (less accurate)
train_epochs=300  # instead of 1000

# Or enable GPU if available
device='cuda'  # instead of 'cpu'
```

### Problem: Output colors don't match palette

**Solution**: Increase training or adjust palette size

```python
train_epochs=1500  # More training
num_distinct=5     # Simpler palette
```

### Problem: "No such file or directory"

**Solution**: Check your working directory

```python
import os
print(os.getcwd())  # Should show music visualizer directory
print(os.listdir('originalImages'))  # Should list your images
```

## Key Parameters to Try

```python
results = imageTriangulation.pipeline_with_cnn(
    source_image_path='originalImages/spiderman.jpg',
    target_image_path='hybridTheory.jpeg',
    
    # Triangulation parameters
    threshold=50,              # Edge detection threshold
    density_reduction=60,      # More = fewer triangles
    
    # Palette parameters
    num_clusters=25,           # K-Means clusters
    num_distinct=10,           # Final palette size
    
    # Training parameters
    train_epochs=1000,         # 300-2000
    
    # Output
    save_model_path='models/my_model.pth',  # Optional
    save_output=True,          # Save images?
    device='cpu'               # or 'cuda'
)
```

## Understanding Training Output

```
Epoch 50/500 - Total: 0.234567, Histogram: 0.150000, Nearest: 0.063333, Smoothness: 0.021234
```

- **Total Loss**: Overall loss (should decrease over time)
- **Histogram Loss**: How well output colors match target distribution
- **Nearest Loss**: How close outputs are to palette colors
- **Smoothness Loss**: Stability of color transformations

✓ **Good training**: Loss decreases consistently  
✗ **Bad training**: Loss stays same or increases

## Results Location

Outputs are saved to:

```
triangulatedImages/
├── cnn_triangulation.png      # Main result
└── (other method results)

models/
└── my_model.pth               # Trained model (optional)
```

## Next Steps

1. **Try different image pairs**
   ```python
   # Create interesting combinations
   source='originalImages/A.jpg'
   target='originalImages/B.jpg'
   ```

2. **Compare methods side-by-side**
   ```python
   # See how CNN compares to simpler methods
   CNN.compare_methods(S, triangles, image_orig, ...)
   ```

3. **Reuse trained models**
   ```python
   # Apply saved model (much faster)
   use_pretrained_model='models/my_model.pth'
   ```

4. **Experiment with parameters**
   ```python
   # Try different training configurations
   train_epochs=500
   num_distinct=5
   density_reduction=30
   ```

## Performance Tips

### Faster Training
- Use GPU: `device='cuda'` (if available)
- Smaller palette: `num_distinct=5`
- Fewer epochs: `train_epochs=300`
- Larger batch size: (set in train function)

### Better Quality
- More epochs: `train_epochs=2000`
- More clusters: `num_clusters=30`
- More distinct colors: `num_distinct=15`
- Finer triangulation: `density_reduction=30`

### Balanced
- Default settings in examples
- ~10-15 minutes training time on CPU
- Good quality results
- Reasonable computation

## Common Questions

### Q: Can I use my own images?

**A**: Yes! Just specify the path:
```python
imageTriangulation.pipeline_with_cnn(
    source_image_path='path/to/your/image.jpg',
    target_image_path='path/to/style/image.jpg'
)
```

### Q: How long does training take?

**A**: 
- CPU: 5-20 minutes (depends on parameters)
- GPU: 10-60 seconds

### Q: Can I stop training early?

**A**: Yes, loss usually stabilizes by epoch 500-800. Later epochs give diminishing returns.

### Q: Can I use the model on different images?

**A**: Yes! Once trained on two images, the model learns a general color mapping. You can apply it to other source images with the same target style.

### Q: How much disk space do I need?

**A**: 
- Model files: ~2MB each
- Training images: normal image size
- Results: ~image size

## File Locations

```
/Users/ahmadwali04/Desktop/personal/Projects/musicVisualizer/

Main files:
├── CNN.py                    ← Neural network implementation
├── colour.py                 ← Color palette functions
├── imageTriangulation.py     ← Triangulation functions
├── example_cnn_usage.py      ← Run these examples

Documentation:
├── README_CNN.md             ← Full documentation
├── CNN_IMPLEMENTATION_SUMMARY.md  ← Technical details
└── QUICK_START.md           ← This file

Data directories:
├── originalImages/           ← Your images here
├── triangulatedImages/       ← Results saved here
└── models/                   ← Trained models saved here
```

## One-Liner to Get Started

```bash
cd /Users/ahmadwali04/Desktop/personal/Projects/musicVisualizer && \
source .venv/bin/activate && \
python example_cnn_usage.py
```

Then select option 4 for a quick palette preview, then option 1 for full training!

---

**Ready to start?** Run `python example_cnn_usage.py` now! 🚀
