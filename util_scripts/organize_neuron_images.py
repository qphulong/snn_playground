"""
Reorganize neuron visualization images by neuron index.

This script searches for PNG image files generated during training
(e.g., weight visualizations per neuron across epochs), extracts
the epoch number and neuron ID from each filename, and copies the
images into a new directory structure grouped by neuron.

Input pattern:
    training/two_layers/vizs/epoch_*/weights_per_neuron_sample*_neuron*.png

For each matched file:
    - Extracts:
        * epoch number (e.g., 47)
        * neuron ID (e.g., 0033)
    - Creates a destination folder per neuron:
        images/neuron<neuron_id>/
    - Renames the file to:
        neuron<neuron_id>_epoch<epoch>.png
    - Copies the file into the corresponding neuron folder

This makes it easier to inspect how each neuron evolves over epochs.

Dependencies:
    - os
    - re
    - glob
    - shutil

Usage:
    Run the script from the project root directory:
        python3 util_scripts/organize_neuron_images.py
"""

import os
import re
import glob
import shutil

pattern = "training/two_layers/vizs/epoch_*/weights_per_neuron_sample*_neuron*.png"
regex = re.compile(r'epoch_(\d+).*neuron(\d+)\.png')

for filepath in glob.glob(pattern):
    match = regex.search(filepath)
    if not match:
        continue

    epoch = match.group(1)     # "47"
    neuron = match.group(2)    # "0033"

    # destination folder per neuron
    dst_dir = os.path.join("images", f"neuron{neuron}")
    os.makedirs(dst_dir, exist_ok=True)

    # new filename
    new_name = f"neuron{neuron}_epoch{epoch}.png"
    dst_path = os.path.join(dst_dir, new_name)

    shutil.copy2(filepath, dst_path)

print("Done.")