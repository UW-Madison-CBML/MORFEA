from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import random
from pathlib import Path

app = Flask(__name__)

# Configuration
DATASET_DIR = Path("embryo_dataset")
TRUE_IMAGES_FILE = Path("true_images.txt")

# Create necessary directories
DATASET_DIR.mkdir(exist_ok=True)

# Image extensions to look for
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}


def get_all_images():
    """Recursively find all images in the dataset directory."""
    images = []
    if DATASET_DIR.exists():
        for root, dirs, files in os.walk(DATASET_DIR):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, file))
    return images


def get_next_image():
    """Get a random unlabeled image from the dataset."""
    all_images = get_all_images()

    if not all_images:
        return None

    # Randomly select an image
    return random.choice(all_images)


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/next-image')
def next_image():
    """Get the next image to label."""
    image_path = get_next_image()

    if not image_path:
        return jsonify({'error': 'No images found in embryo_dataset'}), 404

    return jsonify({
        'image_path': image_path,
        'image_url': f'/image/{image_path}'
    })


@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve an image file."""
    full_path = Path(image_path)

    # Security check: ensure the path is within embryo_dataset
    try:
        full_path.resolve().relative_to(DATASET_DIR.resolve())
    except ValueError:
        return "Access denied", 403

    return send_from_directory(DATASET_DIR, image_path)


@app.route('/api/label', methods=['POST'])
def label_image():
    """Label an image as true or false."""
    data = request.json
    image_path = data.get('image_path')
    label = data.get('label')  # 'true' or 'false'

    if not image_path or label not in ['true', 'false']:
        return jsonify({'error': 'Invalid request'}), 400

    source_path = Path(image_path)

    if not source_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    # If label is 'true', append to the text file
    if label == 'true':
        with open(TRUE_IMAGES_FILE, 'a') as f:
            f.write(f"{image_path}\n")

    return jsonify({
        'success': True,
        'label': label,
        'image_path': image_path
    })


@app.route('/api/stats')
def get_stats():
    """Get labeling statistics."""
    true_count = 0
    if TRUE_IMAGES_FILE.exists():
        with open(TRUE_IMAGES_FILE, 'r') as f:
            true_count = len(f.readlines())

    remaining = len(get_all_images())

    return jsonify({
        'true': true_count,
        'remaining': remaining,
        'total': true_count + remaining
    })


if __name__ == '__main__':
    print("Starting Embryo Image Labeler Server...")
    print(f"Dataset directory: {DATASET_DIR.absolute()}")
    print(f"True images file: {TRUE_IMAGES_FILE.absolute()}")
    print("Server running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
