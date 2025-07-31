import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Global variables for ROI selection
roi_points = []
roi_image = None
selecting_roi = False
selection_done = False

def find_image(base_name, folder="Images"):
    formats = ['.jpg', '.png', '.jfif']
    for fmt in formats:
        file_path = os.path.join(folder, base_name + fmt)
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(f"No image file found for {base_name} in {folder}")

def load_and_convert_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded")
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_blur(gray_image):
    return cv2.GaussianBlur(gray_image, (7, 7), 0)

def apply_threshold(blurred_image):
    return cv2.adaptiveThreshold(blurred_image, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

def remove_lines(thresh_image):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    
    # Remove horizontal lines
    no_horizontal = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # Remove vertical lines
    no_vertical = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Subtract lines from the threshold image
    clean = cv2.subtract(thresh_image, no_horizontal)
    clean = cv2.subtract(clean, no_vertical)
    
    return clean

def apply_morphology(thresh_image):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

def create_circular_mask(image):
    mask = np.zeros_like(image)
    height, width = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1], width - center[0], height - center[1]) - 10
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(image, mask)

def is_circular(contour, min_circularity=0.6):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity > min_circularity

def find_colonies(masked_image, min_area=10):
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and is_circular(cnt)]

def draw_results(original_image, contours):
    output = original_image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
    return output

def plot_results(original_image, masked_image, output_image, colony_count, roi_result=None):
    if roi_result is None:
        plt.figure(figsize=(12, 6))
    
        # Original image without boundaries
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        # Create masked binary image with boundaries
        masked_with_boundaries = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked_with_boundaries, contours, -1, (255, 0, 0), 1)

        plt.subplot(1, 3, 2)
        plt.imshow(masked_with_boundaries)
        plt.title("Masked Binary with Boundaries")
        plt.axis("off")

        # Create output image with green boundaries
        output_with_boundaries = original_image.copy()
        cv2.drawContours(output_with_boundaries, contours, -1, (0, 255, 0), 1)
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(output_with_boundaries, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Samples: {colony_count}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(15, 6))
        # # Full image results
        # plt.subplot(2, 3, 1)
        # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        # plt.title("Original")
        # plt.axis("off")

        # plt.subplot(2, 3, 2)
        # plt.imshow(masked_image, cmap='gray')
        # plt.title("Full Image Binary")
        # plt.axis("off")

        # plt.subplot(2, 3, 3)
        # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        # plt.title(f"Full Image Detected: {colony_count}")
        # plt.axis("off")

        # ROI results
        roi_img, roi_mask, roi_out, roi_count = roi_result
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        plt.title("ROI")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(roi_mask, cmap='gray')
        plt.title("ROI Binary")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(roi_out, cv2.COLOR_BGR2RGB))
        plt.title(f"ROI Detected: {roi_count}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def mouse_callback(event, x, y, flags, param):
    global roi_points, selecting_roi, selection_done, roi_image, display_image, scale_factor
    
    # Convert coordinates to original image scale
    orig_x = int(x / scale_factor)
    orig_y = int(y / scale_factor)
    
    if event == cv2.EVENT_LBUTTONDOWN and not selecting_roi:
        roi_points = [(orig_x, orig_y)]
        selecting_roi = True
        
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        center = roi_points[0]
        radius = int(((orig_x - center[0])**2 + (orig_y - center[1])**2)**0.5)
        display_image = display_buffer.copy()
        cv2.circle(display_image, 
                  (int(center[0] * scale_factor), int(center[1] * scale_factor)), 
                  int(radius * scale_factor), 
                  (0, 255, 0), 2)
        
    elif event == cv2.EVENT_LBUTTONUP and selecting_roi:
        center = roi_points[0]
        radius = int(((orig_x - center[0])**2 + (orig_y - center[1])**2)**0.5)
        roi_points.append(radius)
        selecting_roi = False
        selection_done = True

def adjust_circle(image, initial_center, initial_radius):
    window_name = 'Adjust Circle'
    center = list(initial_center)
    radius = initial_radius
    dragging = False
    drag_start = None
    
    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, drag_start, center, radius
        
        # Show distance from center and edge when mouse moves
        dist_from_center = ((x - center[0])**2 + (y - center[1])**2)**0.5
        near_edge = abs(dist_from_center - radius) < 10
        near_center = abs(x - center[0]) < 10 and abs(y - center[1]) < 10
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if near_edge:
                dragging = 'radius'
            elif near_center:
                dragging = 'position'
            drag_start = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            if dragging == 'radius':
                radius = int(((x - center[0])**2 + (y - center[1])**2)**0.5)
            elif dragging == 'position':
                dx = x - drag_start[0]
                dy = y - drag_start[1]
                center[0] += dx
                center[1] += dy
                drag_start = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    
    while True:
        # Create fresh copy of image for each frame
        adj_image = image.copy()
        
        # Always draw the circle and center point
        cv2.circle(adj_image, (center[0], center[1]), radius, (0, 255, 0), 2)
        cv2.circle(adj_image, (center[0], center[1]), 5, (255, 0, 0), -1)
        
        # Add instructions on image
        text_color = (0, 255, 255)
        cv2.putText(adj_image, "Drag center (blue) to move", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(adj_image, "Drag edge (green) to resize", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(adj_image, "Press ENTER to confirm", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.imshow(window_name, adj_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            cv2.destroyWindow(window_name)
            return (center[0], center[1]), radius
        elif key == 27:  # Escape key
            cv2.destroyWindow(window_name)
            return None

def get_roi(image):
    global roi_image, roi_points, selection_done, display_image, display_buffer, scale_factor
    
    # Reset selection state
    roi_points = []
    selection_done = False
    
    # Resize image for faster display
    max_display_width = 800
    scale_factor = max_display_width / image.shape[1]
    display_image = resize_with_aspect_ratio(image, width=max_display_width)
    display_buffer = display_image.copy()
    
    cv2.namedWindow('Select Circular ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select Circular ROI', display_image.shape[1], display_image.shape[0])
    cv2.setMouseCallback('Select Circular ROI', mouse_callback)
    
    print("Click and drag to select circular ROI, Press ESC to cancel")
    
    while not selection_done:
        cv2.imshow('Select Circular ROI', display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow('Select Circular ROI')
            return None
    
    cv2.destroyWindow('Select Circular ROI')
    center_x, center_y = roi_points[0]
    radius = roi_points[1]
    
    # Add adjustment phase
    print("Adjust circle position and size:")
    print("- Drag center point to move")
    print("- Drag circle edge to resize")
    print("- Press ENTER to confirm")
    print("- Press ESC to cancel")
    
    adjusted = adjust_circle(image, (center_x, center_y), radius)
    if adjusted is None:
        return None
        
    return (adjusted[0][0], adjusted[0][1], adjusted[1])

def process_roi(image, roi):
    if roi is None:
        return image
        
    center_x, center_y, radius = roi
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    if len(image.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return cv2.bitwise_and(image, mask)

def main():
    try:
        image_path = find_image("detection3")
        img, gray = load_and_convert_image(image_path)
        
        # Process full image first
        blurred = apply_blur(gray)
        thresh = apply_threshold(blurred)
        no_lines = remove_lines(thresh)
        morphed = apply_morphology(no_lines)
        masked = create_circular_mask(morphed)
        colonies = find_colonies(masked)
        output = draw_results(img, colonies)
        full_count = len(colonies)
        
        # Get ROI from user
        roi = get_roi(img)
        if roi is None:
            print("ROI selection cancelled")
            plot_results(img, masked, output, full_count)
            return
        
        # Process the ROI
        img_roi = process_roi(img, roi)
        gray_roi = process_roi(gray, roi)
        
        # Apply processing on ROI
        blurred_roi = apply_blur(gray_roi)
        thresh_roi = apply_threshold(blurred_roi)
        no_lines_roi = remove_lines(thresh_roi)
        morphed_roi = apply_morphology(no_lines_roi)
        masked_roi = create_circular_mask(morphed_roi)
        
        colonies_roi = find_colonies(masked_roi)
        output_roi = draw_results(img_roi, colonies_roi)
        
        print(f"Number of colonies detected - Full: {full_count}, ROI: {len(colonies_roi)}")
        roi_result = (img_roi, masked_roi, output_roi, len(colonies_roi))
        plot_results(img, masked, output, full_count, roi_result)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        exit(1)

if __name__ == "__main__":
    main()
