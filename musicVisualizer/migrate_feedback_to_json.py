"""
migrate_feedback_to_json.py - Convert old filename-based feedback to JSON

This script migrates feedback from the old system (encoded in image filenames)
to the new JSON-based system for TensorBoard compatibility.

Usage:
    python migrate_feedback_to_json.py
"""

import os
import glob
import re
import json
from datetime import datetime
from pathlib import Path


def parse_feedback_filename(filename):
    """
    Parse feedback from old filename format.

    Format: ABC_5739826400_5739826400.png
            └─ hex prefix
               └─ frequency scores
                  └─ placement scores

    Returns:
        dict or None: Feedback data if valid, None otherwise
    """
    # Remove extension
    name = os.path.splitext(os.path.basename(filename))[0]

    # Match pattern: HEX_DIGITS_DIGITS
    match = re.match(r'([0-9A-F]{3})_(\d{10})_(\d{10})', name)

    if not match:
        return None

    hex_prefix, freq_str, place_str = match.groups()

    # Parse scores
    try:
        freq_scores = [int(d) for d in freq_str]
        place_scores = [int(d) for d in place_str]
    except (ValueError, IndexError):
        return None

    # Validate score counts
    if len(freq_scores) != 10 or len(place_scores) != 10:
        return None

    # Estimate session number from hex prefix (incremental counter)
    session_number = int(hex_prefix, 16)

    return {
        'session_name': f'migrated_session_{session_number:04d}',
        'timestamp': None,  # Unknown, will use file modification time
        'frequency_scores': freq_scores,
        'placement_scores': place_scores,
        'palette_rgb': None,  # Not stored in old format
        'palette_lab': None,  # Not stored in old format
        'migrated_from': filename
    }


def migrate_feedback(old_dir='trainingData', new_dir='feedback_data'):
    """
    Migrate all feedback from old system to new JSON format.

    Args:
        old_dir: Directory containing old feedback images
        new_dir: Directory for new JSON feedback files
    """
    # Create new directory
    Path(new_dir).mkdir(exist_ok=True)

    # Find all old feedback images
    old_files = sorted(glob.glob(os.path.join(old_dir, '*.png')))

    if not old_files:
        print(f"\nNo files found in {old_dir}/")
        print("Nothing to migrate.")
        return

    print(f"\nFound {len(old_files)} feedback files to migrate")
    print("="*70)

    migrated_count = 0
    skipped_count = 0

    for filepath in old_files:
        # Parse feedback
        feedback = parse_feedback_filename(filepath)

        if feedback is None:
            print(f"⚠️  Skipping invalid filename: {os.path.basename(filepath)}")
            skipped_count += 1
            continue

        # Use file modification time as timestamp
        try:
            mtime = os.path.getmtime(filepath)
            feedback['timestamp'] = datetime.fromtimestamp(mtime).isoformat()
        except OSError:
            feedback['timestamp'] = datetime.now().isoformat()

        # Save as JSON
        json_filename = os.path.join(new_dir, f"{feedback['session_name']}.json")

        # Check if already exists
        if os.path.exists(json_filename):
            print(f"⚠️  Skipping (already exists): {feedback['session_name']}.json")
            skipped_count += 1
            continue

        try:
            with open(json_filename, 'w') as f:
                json.dump(feedback, f, indent=2)

            print(f"✓ Migrated: {os.path.basename(filepath)} → {feedback['session_name']}.json")
            migrated_count += 1
        except IOError as e:
            print(f"❌ Error saving {feedback['session_name']}.json: {e}")
            skipped_count += 1

    print("="*70)
    print(f"\n✓ Migration complete!")
    print(f"  Migrated: {migrated_count} files")
    print(f"  Skipped:  {skipped_count} files")
    print(f"\nOld feedback: {old_dir}/")
    print(f"New feedback: {new_dir}/")

    if migrated_count > 0:
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Verify migration: Check files in {new_dir}/")
        print(f"   2. Test system: Run example_cnn_usage.py")
        print(f"   3. Backup old data: cp -r {old_dir}/ {old_dir}_backup/")
        print(f"   4. Clean up (optional): rm -rf {old_dir}/")


def verify_migration(feedback_dir='feedback_data'):
    """
    Verify migrated JSON files are valid.

    Args:
        feedback_dir: Directory containing feedback JSON files
    """
    feedback_dir = Path(feedback_dir)

    if not feedback_dir.exists():
        print(f"Directory not found: {feedback_dir}")
        return

    json_files = list(feedback_dir.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {feedback_dir}/")
        return

    print(f"\nVerifying {len(json_files)} feedback files...")
    print("="*70)

    valid_count = 0
    invalid_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check required fields
            required_fields = ['session_name', 'timestamp', 'frequency_scores', 'placement_scores']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                print(f"❌ Invalid (missing fields): {json_file.name}")
                print(f"   Missing: {', '.join(missing_fields)}")
                invalid_count += 1
                continue

            # Check score counts
            if len(data['frequency_scores']) != 10:
                print(f"❌ Invalid (frequency_scores != 10): {json_file.name}")
                invalid_count += 1
                continue

            if len(data['placement_scores']) != 10:
                print(f"❌ Invalid (placement_scores != 10): {json_file.name}")
                invalid_count += 1
                continue

            # Check score ranges
            all_scores = data['frequency_scores'] + data['placement_scores']
            if not all(0 <= score <= 9 for score in all_scores):
                print(f"⚠️  Warning (scores out of range 0-9): {json_file.name}")

            print(f"✓ Valid: {json_file.name}")
            valid_count += 1

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {json_file.name}")
            print(f"   Error: {e}")
            invalid_count += 1
        except Exception as e:
            print(f"❌ Error reading: {json_file.name}")
            print(f"   Error: {e}")
            invalid_count += 1

    print("="*70)
    print(f"\nVerification complete!")
    print(f"  Valid:   {valid_count} files")
    print(f"  Invalid: {invalid_count} files")

    if invalid_count > 0:
        print(f"\n⚠️  Please fix invalid files before using the system")
    else:
        print(f"\n✓ All files are valid!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("FEEDBACK MIGRATION TOOL")
    print("="*70)
    print("\nThis will convert old filename-based feedback to JSON format")
    print("Old format: trainingData/ABC_5739826400_5739826400.png")
    print("New format: feedback_data/migrated_session_0000.json")
    print("="*70)

    # Check if old directory exists
    if not os.path.exists('trainingData'):
        print(f"\n⚠️  No 'trainingData/' directory found")
        print(f"   Nothing to migrate")
        print(f"\n💡 If you have old feedback elsewhere, edit this script:")
        print(f"   Change 'trainingData' to your directory path")
    else:
        # Ask for confirmation
        response = input("\nProceed with migration? (y/n): ").strip().lower()

        if response == 'y':
            migrate_feedback()

            # Verify migration
            verify_response = input("\nVerify migrated files? (y/n): ").strip().lower()
            if verify_response == 'y':
                verify_migration()
        else:
            print("\nMigration cancelled")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
