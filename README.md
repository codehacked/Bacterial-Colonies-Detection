# 🧫 Bacterial Colony Detection and ROI Analysis Tool

This tool detects bacterial colonies in petri dish images and allows users to interactively select and analyze a **Region of Interest (ROI)** using a circular selection method. The program applies several image preprocessing techniques and contour analysis to count circular bacterial colonies within the image or a user-defined ROI.

---

## 🔧 Features

- Load images from a predefined folder and support multiple image formats.
- Preprocess the image using Gaussian blur and adaptive thresholding.
- Remove grid lines (horizontal & vertical) using morphological operations.
- Automatically mask out the circular petri dish region.
- Detect circular colonies using contour analysis and circularity filtering.
- Interactive selection of circular ROI (with resizing and repositioning).
- Display original, masked, and result images with bounding contours.
- Show colony count for both the full image and the ROI.

---

## 📁 Folder Structure

```
project/
├── Images/
│   └── detection3.jpg        # Place your image(s) here
├── script.py                 # Main script for detection and ROI
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 📦 Installation

1. Clone or download this repository.
2. Place your input images inside the `Images/` folder.
3. Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Use

1. Run the script:

```bash
python script.py
```

2. On launch:
   - Click and drag on the image to draw a **circular ROI**.
   - Use your mouse to **move** or **resize** the circle.
   - Press **Enter** to confirm or **Esc** to cancel ROI selection.

---

## 🧠 Pipeline Summary

### 1. **Image Loading**
- Script searches for an image named `detection3` in the `Images/` folder with `.jpg`, `.png`, or `.jfif` extension.

### 2. **Preprocessing**
- Converts image to grayscale.
- Applies Gaussian blur to reduce noise.
- Uses adaptive thresholding to generate a binary mask.

### 3. **Grid/Line Removal**
- Applies morphological operations to remove horizontal and vertical grid lines.

### 4. **Cleaning**
- Morphological closing and opening operations clean up small noise and fill holes.

### 5. **Circular Masking**
- A circular mask is applied to isolate the petri dish region, ignoring irrelevant borders.

### 6. **Contour Detection**
- Detects contours and filters:
  - Based on **area**.
  - Based on **circularity** (threshold ≥ 0.6).

### 7. **Interactive ROI Selection**
- After full-image analysis, the user can select a circular region.
- The program repeats the pipeline within the selected ROI.

### 8. **Result Visualization**
- Displays:
  - Original image with marked colonies.
  - Binary mask with contours.
  - ROI-based detection.
- Displays colony counts both globally and within ROI.

---

## 🖼️ Sample Output

- ✅ Original Petri Dish Image
- 🧠 Binary Image with Colony Masks
- 🔍 ROI with Detected Colonies
- 📊 Colony Counts: Global + ROI

---

## ❗ Notes

- Optimized for petri dish images with clear contrast between colonies and background.
- ROI tool helps target specific dish areas, even with noise or grid interference.
- You can adjust detection thresholds and morphology settings in the script if needed.

---

## 📜 License

This project is intended for academic and research purposes. For commercial usage, please contact the author.
