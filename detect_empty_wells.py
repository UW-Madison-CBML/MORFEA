"""
Empty well detection system for embryo images.
Detects empty wells using a combination of:
1. Heuristic features (entropy, contrast, well region intensity)
2. Optional trained neural network classifier for refinement

Usage:
    detector = EmptyWellDetector(model_path="empty_well_model.pth")
    prob = detector.predict(image_path)

    Or batch process:
    probs = detector.predict_batch(image_paths, batch_size=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
import cv2
from typing import Union, List, Tuple
from scipy import stats
from tqdm import tqdm
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EmptyWellClassifier(nn.Module):
    """Lightweight CNN for classifying empty vs. non-empty wells"""

    def __init__(self, input_size: int = 128):
        super().__init__()
        self.input_size = input_size

        # Lightweight conv layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Calculate flattened size
        self.flat_size = 128 * (input_size // 8) * (input_size // 8)

        # Classifier head
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logit for empty well probability"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class EmptyWellDetector:
    """Main detector combining heuristics and neural network"""

    def __init__(
        self,
        model_path: Union[str, Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_heuristics: bool = True,
        heuristic_weight: float = 0.3,
        use_gpu: bool = True,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to trained classifier (optional)
            device: Device to use ('cuda' or 'cpu')
            use_heuristics: Whether to use heuristic features
            heuristic_weight: Weight for heuristic score (0-1, blended with model)
            use_gpu: Enable GPU processing for images
        """
        self.device = device if use_gpu else "cpu"
        self.use_heuristics = use_heuristics
        self.heuristic_weight = heuristic_weight
        self.model = None

        if model_path and Path(model_path).exists():
            self.model = EmptyWellClassifier().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load grayscale image as float32 normalized [0, 1]"""
        try:
            img = np.array(Image.open(image_path), dtype=np.float32)
            if img.ndim == 3:
                # Convert RGB to grayscale if needed
                img = np.mean(img, axis=2)
            img = np.clip(img / 255.0, 0, 1)
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def _detect_well_circle(self, img: np.ndarray) -> Tuple[int, int, int]:
        """
        Detect the well circle using Hough Circle Detection.
        Returns: (center_x, center_y, radius)
        """
        # Normalize to 0-255 for processing
        img_8bit = (img * 255).astype(np.uint8)

        # Detect circles (well region is typically a bright circle)
        circles = cv2.HoughCircles(
            img_8bit,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=30,
            param2=20,
            minRadius=100,
            maxRadius=300,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return the most prominent circle (first one)
            x, y, r = circles[0][0]
            return int(x), int(y), int(r)
        else:
            # Fallback: assume circle is at center
            h, w = img.shape
            center_x, center_y = w // 2, h // 2
            radius = min(h, w) // 3
            return center_x, center_y, radius

    def _extract_well_region(
        self, img: np.ndarray, cx: int, cy: int, r: int
    ) -> np.ndarray:
        """Extract circular region of interest from image"""
        h, w = img.shape
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

        # Extract well region
        well_region = np.zeros_like(img)
        well_region[mask] = img[mask]

        # Crop to bounding box
        y_min, y_max = max(0, cy - r), min(h, cy + r)
        x_min, x_max = max(0, cx - r), min(w, cx + r)

        return well_region[y_min:y_max, x_min:x_max]

    def _compute_heuristic_score(self, img: np.ndarray) -> float:
        """
        Compute heuristic empty well probability.

        Empty wells tend to have:
        - Low entropy (uniform gray)
        - Low contrast
        - Specific intensity range (mid-gray)

        Returns: probability in [0, 1]
        """
        # Detect well circle
        cx, cy, r = self._detect_well_circle(img)

        # Extract well region
        well_img = self._extract_well_region(img, cx, cy, r)
        well_pixels = well_img[well_img > 0.01]  # Exclude black regions

        if len(well_pixels) == 0:
            return 1.0  # Likely empty if no bright pixels

        scores = []

        # 1. Entropy score (empty = low entropy)
        hist, _ = np.histogram(well_pixels, bins=32, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        entropy_score = 1.0 - min(entropy / 8.0, 1.0)  # Normalize to [0, 1]
        scores.append(entropy_score)

        # 2. Contrast score (empty = low contrast)
        contrast = np.std(well_pixels)
        contrast_score = 1.0 - min(contrast, 1.0)
        scores.append(contrast_score)

        # 3. Intensity score (empty wells are mid-gray ~0.4-0.6)
        mean_intensity = np.mean(well_pixels)
        ideal_empty = 0.5
        intensity_score = 1.0 - abs(mean_intensity - ideal_empty) * 2.0
        intensity_score = max(0, intensity_score)
        scores.append(intensity_score)

        # 4. Peak frequency score (empty = concentrated at one value)
        peak_freq = np.max(hist)
        peak_score = peak_freq  # Higher peak = more uniform
        scores.append(peak_score)

        # Weighted combination
        heuristic_prob = (
            scores[0] * 0.3 +  # entropy
            scores[1] * 0.3 +  # contrast
            scores[2] * 0.2 +  # intensity
            scores[3] * 0.2    # peak frequency
        )

        return np.clip(heuristic_prob, 0, 1)

    def _preprocess_for_model(self, img: np.ndarray, size: int = 128) -> torch.Tensor:
        """Preprocess image for neural network"""
        # Resize to model input size
        if img.shape != (size, size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        # Convert to tensor
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_path: Union[str, Path]) -> float:
        """
        Predict empty well probability for a single image.

        Returns: probability in [0, 1]
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = self._load_image(image_path)
        if img is None:
            return 0.5  # Return neutral score on error

        # Get heuristic score
        if self.use_heuristics:
            heuristic_score = self._compute_heuristic_score(img)
        else:
            heuristic_score = 0.5

        # Get model score if available
        if self.model is not None:
            with torch.no_grad():
                tensor = self._preprocess_for_model(img)
                model_score = self.model(tensor).item()

            # Blend scores
            final_score = (
                self.heuristic_weight * heuristic_score +
                (1 - self.heuristic_weight) * model_score
            )
        else:
            final_score = heuristic_score

        return np.clip(final_score, 0, 1)

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[float]:
        """
        Predict empty well probabilities for multiple images.

        Returns: list of probabilities
        """
        results = []

        iterator = tqdm(image_paths, disable=not show_progress)
        for image_path in iterator:
            try:
                prob = self.predict(image_path)
                results.append(prob)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(0.5)  # Default to neutral on error

        return results

    def predict_with_confidence(
        self, image_path: Union[str, Path]
    ) -> Tuple[float, str]:
        """
        Predict empty well with confidence level.

        Returns: (probability, confidence_level)
        """
        prob = self.predict(image_path)

        if prob > 0.7:
            confidence = "HIGH"
        elif prob > 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return prob, confidence


def create_training_dataset(
    dataset_root: Union[str, Path],
    output_csv: Union[str, Path],
    prob_threshold: float = 0.6,
    model_path: Union[str, Path] = None,
) -> None:
    """
    Scan dataset and label images as empty_well=True/False.
    Adds a new column to the CSV output.

    Args:
        dataset_root: Root directory of embryo dataset
        output_csv: Path to output CSV with empty_well labels
        prob_threshold: Probability threshold for empty classification
        model_path: Optional path to trained model
    """
    detector = EmptyWellDetector(model_path=model_path)

    dataset_root = Path(dataset_root)
    cell_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    rows = []

    for cell_dir in tqdm(cell_dirs, desc="Processing cells"):
        # Find all image files
        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        images = []
        for ext in image_exts:
            images.extend(cell_dir.glob(ext))

        images = sorted(set(images))

        for img_path in images:
            if not img_path.stat().st_size > 0:
                continue

            try:
                prob = detector.predict(img_path)
                is_empty = prob >= prob_threshold

                rows.append({
                    "cell_id": cell_dir.name,
                    "image_path": str(img_path),
                    "empty_prob": prob,
                    "empty_well": is_empty,
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                rows.append({
                    "cell_id": cell_dir.name,
                    "image_path": str(img_path),
                    "empty_prob": 0.5,
                    "empty_well": False,
                })

    # Save results
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} results to {output_csv}")

    # Print statistics
    num_empty = df["empty_well"].sum()
    num_total = len(df)
    print(f"Empty wells detected: {num_empty}/{num_total} ({100*num_empty/num_total:.1f}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect_empty_wells.py <image_or_directory>")
        print("       python detect_empty_wells.py --batch <directory>")
        sys.exit(1)

    if sys.argv[1] == "--batch" and len(sys.argv) > 2:
        dataset_root = Path(sys.argv[2])
        model_path = sys.argv[3] if len(sys.argv) > 3 else None
        create_training_dataset(
            dataset_root,
            output_csv="empty_wells.csv",
            prob_threshold=0.6,
            model_path=model_path,
        )
    else:
        detector = EmptyWellDetector()
        image_path = Path(sys.argv[1])

        if image_path.is_file():
            prob = detector.predict(image_path)
            print(f"Empty well probability: {prob:.4f}")
        elif image_path.is_dir():
            image_exts = ("*.jpg", "*.jpeg", "*.png")
            images = []
            for ext in image_exts:
                images.extend(image_path.rglob(ext))

            probs = detector.predict_batch(images)
            for img, prob in zip(images, probs):
                print(f"{img.name}: {prob:.4f}")
