import cv2

# 1. Load the image
image_path = 'circle_image.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from '{image_path}'. Check the path.")
    exit()

# Create a copy so we keep the clean original untouched
output_image = image.copy()

# 2. Preprocess to grayscale and binary threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Using OTSU thresholding to automatically calculate the best contrast limit
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circle_count = 0

for contour in contours:
    # Filter out tiny pixel noise
    if cv2.contourArea(contour) > 100: 
        circle_count += 1
        
        # 4. Calculate the minimum enclosing circle properties
        (x, y), pixel_radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(pixel_radius)
        
        # 5. Visual Overlays
        # Draw the outer circle edge (Green line, thickness of 3 pixels)
        cv2.circle(output_image, center, radius, (0, 255, 0), 3)
        # Draw the absolute center point (Red dot)
        cv2.circle(output_image, center, 5, (0, 0, 255), -1)
        
        # Overlay the measurement text onto the image window
        text = f"Radius: {pixel_radius:.1f}px"
        cv2.putText(output_image, text, (center[0] - 60, center[1] - (radius + 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Print logs to terminal
        print(f"--- Circle #{circle_count} Detected ---")
        print(f"Center position: {center}")
        print(f"Radius size: {pixel_radius:.2f} pixels\n")

# 6. Display the result in a UI window
cv2.imshow('Detected Circles & Measurements', output_image)

print("Press '0' or any key while clicking on the image window to close it.")
cv2.waitKey(0)  # Keeps window open until you press a key
cv2.destroyAllWindows()
