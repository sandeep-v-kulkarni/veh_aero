import cv2
import numpy as np
from PIL import Image

def create_feature_visualization(image: Image.Image) -> np.ndarray:
    """Create enhanced feature visualization"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create multi-layer visualization
    viz_image = cv_image.copy()
    
    # Edge detection
    edges = cv2.Canny(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), 100, 200)
    
    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw edges in green
    viz_image[edges > 0] = [0, 255, 0]
    
    # Draw main contour in blue
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(viz_image, [main_contour], -1, (255, 0, 0), 2)
        
        # Draw bounding box in red
        x, y, w, h = cv2.boundingRect(main_contour)
        cv2.rectangle(viz_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)