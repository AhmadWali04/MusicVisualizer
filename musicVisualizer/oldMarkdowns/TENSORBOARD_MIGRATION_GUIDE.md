# TensorBoard Migration Guide

## 🎯 Overview

The feedback system has been completely redesigned to use **TensorBoard** instead of Tkinter, providing:

- ✅ **Cross-platform compatibility** (works on macOS without NSApplication errors)
- ✅ **Browser-based interface** (no native GUI issues)
- ✅ **Rich visualizations** (3D plots, heatmaps, real-time training)
- ✅ **Structured data storage** (JSON instead of filenames)
- ✅ **Industry-standard tooling** (TensorBoard used by PyTorch, TensorFlow, etc.)

---

## 📦 What Changed

### Files Created

1. **`tensorboard_feedback.py`** - New TensorBoard-based feedback system
2. **`migrate_feedback_to_json.py`** - Migration tool for old feedback data
3. **`TENSORBOARD_MIGRATION_GUIDE.md`** - This file

### Files Modified

1. **`CNN.py`**
   - Import changed from `form` to `tensorboard_feedback`
   - `load_previous_feedback()` now reads JSON files
   - `fine_tune_with_feedback()` logs to TensorBoard
   - `save_feedback_image()` deprecated (kept for backward compatibility)

2. **`example_cnn_usage.py`**
   - Import changed from `form` to `tensorboard_feedback`
   - Feedback collection uses TensorBoard visualizations
   - Added `datetime` import for session naming
   - Updated all examples to use new system

3. **`MusicVisualizer/imageTriangulation.py`**
   - Already returns `triangle_colors` (no changes needed)

### Files to Delete (After Migration)

- `form.py` - Old Tkinter-based GUI (replaced by `tensorboard_feedback.py`)
- `trainingData/` - Old feedback images (after migrating to JSON)

---

## 🚀 Quick Start

### 1. Install TensorBoard (if not already installed)

```bash
pip install tensorboard
```

### 2. Migrate Old Feedback (Optional)

If you have existing feedback in `trainingData/`:

```bash
python migrate_feedback_to_json.py
```

This converts old feedback from filenames to JSON format.

### 3. Run Example

```bash
python example_cnn_usage.py
```

Choose **Example 1** from the menu.

### 4. View TensorBoard

When prompted, open TensorBoard in your browser:

```bash
# In a separate terminal
tensorboard --logdir=runs
```

Then navigate to: **http://localhost:6006**

---

## 📊 How It Works

### Old System (Tkinter)

```
1. CNN generates colored triangulation
2. Tkinter GUI shows color swatches and sliders
3. User rates colors (0-9 for frequency and placement)
4. Scores encoded in image filename: ABC_5739826400_5739826400.png
5. CNN.py parses filenames to load feedback
6. Fine-tuning uses parsed feedback
```

**Problems:**
- ❌ Tkinter broken on macOS (NSApplication error)
- ❌ Filename-based storage is fragile
- ❌ No visualization of training progress
- ❌ Limited color usage statistics

### New System (TensorBoard)

```
1. CNN generates colored triangulation
2. TensorBoard displays interactive dashboard:
   - Color palette visualization
   - Color usage statistics (frequency, distance)
   - 3D color distribution (RGB space)
   - Usage heatmap (spatial distribution)
   - Side-by-side comparison (original vs CNN)
3. User views dashboard, then provides ratings via command line
4. Feedback saved as JSON: feedback_data/session_YYYYMMDD_HHMMSS.json
5. CNN.py loads JSON files
6. Fine-tuning logs to TensorBoard (watch loss decrease in real-time)
```

**Benefits:**
- ✅ Works on all platforms (browser-based)
- ✅ Structured JSON storage (queryable, versionable)
- ✅ Rich visualizations (understand color usage)
- ✅ Real-time training monitoring

---

## 📁 Directory Structure

### Before (Old System)

```
project/
├── form.py                    # Tkinter GUI
├── CNN.py
├── example_cnn_usage.py
└── trainingData/              # Feedback encoded in filenames
    ├── 000_5555555555_5555555555.png
    ├── 001_7392648100_7392648100.png
    └── ...
```

### After (New System)

```
project/
├── tensorboard_feedback.py    # NEW: TensorBoard feedback system
├── migrate_feedback_to_json.py # NEW: Migration script
├── CNN.py                     # MODIFIED: Uses JSON feedback
├── example_cnn_usage.py       # MODIFIED: Uses TensorBoard
├── form.py                    # DEPRECATED: Can be deleted
├── feedback_data/             # NEW: JSON feedback storage
│   ├── session_20250204_143022.json
│   ├── session_20250204_150133.json
│   └── ...
└── runs/                      # NEW: TensorBoard logs
    ├── feedback/              # Feedback collection sessions
    │   ├── session_20250204_143022/
    │   └── session_20250204_150133/
    └── fine_tuning/           # Fine-tuning sessions
        └── events.out.tfevents.1234567890.hostname
```

---

## 🎨 TensorBoard Dashboard

### What You'll See

When you open **http://localhost:6006**, you'll see:

#### 1. **SCALARS** Tab
- `Color_Usage/Color_01_Frequency_%` - How much each color is used
- `Color_Usage/Color_01_Avg_Distance` - How closely triangles match palette
- `Fine_Tuning/Feedback_Loss` - Feedback loss during fine-tuning
- `Fine_Tuning/Total_Loss` - Combined loss (feedback + histogram + nearest)

#### 2. **IMAGES** Tab
- `Color_Palette/Target_Palette` - 10 color swatches with RGB/LAB values
- `Comparison/Original_vs_CNN` - Side-by-side original vs CNN output
- `3D_Distribution/RGB_Space` - 3D scatter of colors in RGB space
- `Usage_Heatmap/Spatial_Distribution` - Where each color is used

#### 3. **HISTOGRAMS** Tab
- `Color_Distance_Dist/Color_01` - Distribution of distances to each palette color
- `Fine_Tuning/Output_Channel_0` - RGB channel distributions during training

### How to Use

1. **Before providing feedback:**
   - Look at `Comparison/Original_vs_CNN` to see overall result
   - Check `Color_Usage/` scalars to see which colors are underused
   - View `3D_Distribution/RGB_Space` to see color clustering
   - Study `Usage_Heatmap/` to see spatial distribution

2. **During rating:**
   - Rate **frequency** (0-9): Did the CNN use this color enough?
     - 0 = Never used (increase usage)
     - 5 = Perfect amount
     - 9 = Use much more
   - Rate **placement** (0-9): Was this color used in the right places?
     - 0 = Wrong places
     - 5 = Good placement
     - 9 = Excellent placement

3. **After fine-tuning:**
   - Watch `Fine_Tuning/` losses decrease in real-time
   - Compare multiple sessions side-by-side

---

## 💾 Feedback JSON Format

### Example JSON File

**`feedback_data/session_20250204_143022.json`:**

```json
{
  "session_name": "session_20250204_143022",
  "timestamp": "2025-02-04T14:30:22.123456",
  "frequency_scores": [5, 7, 3, 9, 8, 2, 6, 4, 0, 0],
  "placement_scores": [5, 7, 3, 9, 8, 2, 6, 4, 0, 0],
  "palette_rgb": [
    [245, 230, 220],
    [180, 150, 120],
    [120, 90, 60],
    [200, 170, 140],
    [100, 70, 50],
    [230, 200, 170],
    [150, 120, 90],
    [80, 50, 30],
    [250, 240, 230],
    [60, 40, 20]
  ],
  "palette_lab": [
    [92, 5, 10],
    [67, 8, 20],
    [45, 10, 25],
    [75, 6, 15],
    [38, 12, 28],
    [85, 4, 12],
    [56, 9, 22],
    [28, 15, 30],
    [96, 2, 8],
    [20, 18, 32]
  ]
}
```

### Fields

- `session_name`: Unique session identifier
- `timestamp`: ISO 8601 timestamp
- `frequency_scores`: List of 10 integers (0-9), lightest to darkest
- `placement_scores`: List of 10 integers (0-9), lightest to darkest
- `palette_rgb`: Target palette colors in RGB [0-255]
- `palette_lab`: Target palette colors in LAB space

---

## 🔧 API Changes

### For Users

**OLD (Tkinter):**

```python
import form

# Collect feedback
scores = form.get_user_feedback(palette_rgb, palette_lab)

# Save image with encoded filename
CNN.save_feedback_image(image, palette_rgb, scores)

# Fine-tune
CNN.fine_tune_with_feedback(
    model, source_pixels, target_palette,
    source_pixels_lab, target_palette_lab,
    training_data_dir='trainingData'
)
```

**NEW (TensorBoard):**

```python
import tensorboard_feedback

# Collect feedback via TensorBoard
scores = tensorboard_feedback.get_user_feedback_tensorboard(
    palette_rgb, palette_lab, triangle_colors,
    image_original, image_cnn,
    session_name='my_session'
)

# Feedback automatically saved as JSON
# No need to call save_feedback_image()

# Fine-tune with TensorBoard logging
CNN.fine_tune_with_feedback(
    model, source_pixels, target_palette,
    source_pixels_lab, target_palette_lab,
    feedback_dir='feedback_data',
    log_dir='runs/fine_tuning'
)
```

### For Developers

**Loading Feedback:**

```python
# OLD
feedback_list = CNN.load_previous_feedback('trainingData')

# NEW
feedback_list = CNN.load_previous_feedback('feedback_data')
```

**Return Format (unchanged):**

Both systems return `[(scores, weight), ...]` where:
- `scores`: List of 20 integers [freq0..freq9, place0..place9]
- `weight`: Recency weight (0.3 to 1.0, newer = higher)

---

## 🧪 Testing

### Test the System

```bash
# Test TensorBoard feedback system
python tensorboard_feedback.py
```

This runs a test with dummy data to verify:
- TensorBoard server starts
- Visualizations are logged
- Feedback collection works
- JSON files are saved

### Test Migration

```bash
# Test migration script
python migrate_feedback_to_json.py
```

Follow prompts to migrate old feedback.

### Test Complete Workflow

```bash
# Run Example 1
python example_cnn_usage.py
# Choose option 1

# This will:
# 1. Train CNN (1000 epochs)
# 2. Display TensorBoard dashboard
# 3. Collect feedback via command line
# 4. Fine-tune with feedback (150 epochs)
# 5. Save model and feedback
```

---

## 🐛 Troubleshooting

### TensorBoard Won't Start

**Problem:** Port 6006 is already in use

**Solution:**

```bash
# Find and kill existing TensorBoard process
lsof -i :6006
kill <PID>

# Or use a different port
tensorboard --logdir=runs --port=6007
```

### No Visualizations in TensorBoard

**Problem:** Dashboard is empty

**Solution:**

1. Refresh browser (Ctrl+R or Cmd+R)
2. Check log directory exists: `ls runs/feedback/`
3. Verify files were created: `ls runs/feedback/session_*/`
4. Restart TensorBoard: `pkill -f tensorboard && tensorboard --logdir=runs`

### Feedback JSON Not Created

**Problem:** No JSON files in `feedback_data/`

**Solution:**

1. Check directory exists: `mkdir -p feedback_data`
2. Check write permissions: `ls -la feedback_data/`
3. Look for error messages during feedback collection
4. Verify `tensorboard_feedback.py` is imported correctly

### Migration Script Fails

**Problem:** Migration script reports errors

**Solution:**

1. Check old feedback directory exists: `ls trainingData/`
2. Verify filename format: `ABC_5739826400_5739826400.png`
3. Check for invalid characters in filenames
4. Run verification: `python migrate_feedback_to_json.py` (choose verify option)

### Fine-Tuning Not Using Feedback

**Problem:** Fine-tuning doesn't incorporate feedback

**Solution:**

1. Check feedback files exist: `ls feedback_data/*.json`
2. Verify JSON format: `cat feedback_data/session_*.json`
3. Check function call uses `feedback_dir`: `feedback_dir='feedback_data'`
4. Look for "No previous feedback found" message

---

## 📚 Additional Resources

### TensorBoard Documentation

- Official Guide: https://www.tensorflow.org/tensorboard
- PyTorch Integration: https://pytorch.org/docs/stable/tensorboard.html
- Custom Scalars: https://www.tensorflow.org/tensorboard/scalars_and_keras

### Project Files

- `tensorboard_feedback.py` - TensorBoard feedback system
- `CNN.py` - Neural network color transfer
- `example_cnn_usage.py` - Usage examples
- `colour.py` - Color extraction utilities

### Commands Reference

```bash
# Start TensorBoard
tensorboard --logdir=runs

# Migrate old feedback
python migrate_feedback_to_json.py

# Run examples
python example_cnn_usage.py

# Test TensorBoard system
python tensorboard_feedback.py
```

---

## ✅ Migration Checklist

- [ ] Install TensorBoard: `pip install tensorboard`
- [ ] Run migration script: `python migrate_feedback_to_json.py`
- [ ] Verify migration: Check `feedback_data/*.json`
- [ ] Test TensorBoard: `python tensorboard_feedback.py`
- [ ] Run Example 1: `python example_cnn_usage.py`
- [ ] View dashboard: http://localhost:6006
- [ ] Collect feedback successfully
- [ ] Verify fine-tuning works
- [ ] Backup old data: `cp -r trainingData/ trainingData_backup/`
- [ ] (Optional) Delete old files: `rm -rf trainingData/ form.py`

---

## 🎉 Success Criteria

Your migration is successful when:

✅ TensorBoard starts without errors
✅ Dashboard displays all visualizations
✅ Feedback collection works via command line
✅ JSON files are created in `feedback_data/`
✅ Fine-tuning incorporates feedback
✅ Training progress visible in TensorBoard
✅ Multiple sessions can be compared
✅ No Tkinter errors on macOS

---

## 💬 Support

If you encounter issues:

1. Check this guide's **Troubleshooting** section
2. Verify all files were created/modified correctly
3. Test with dummy data: `python tensorboard_feedback.py`
4. Review TensorBoard logs in `runs/` directory
5. Check JSON files are valid: `python migrate_feedback_to_json.py` (verify option)

---

**🎊 Congratulations!** You've successfully migrated to the TensorBoard-based feedback system. Enjoy rich visualizations and cross-platform compatibility!
