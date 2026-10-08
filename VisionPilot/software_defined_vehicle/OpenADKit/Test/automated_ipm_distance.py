import cv2
import numpy as np

def get_automated_trapezoid(width, height):
    """
    Ultra-strict Ego-Lane isolation to prevent 3D object warping.
    """
    # Keep the vertical horizon slightly below the distant visual clutter
    horizon_y = int(height * 0.62)   
    bottom_y = int(height * 0.95)    
    
    # NARROW THE BASE DRASTICALLY
    # Push the left side past the white line and the median.
    left_bottom_x = int(width * 0.42)  # Shifted heavily right (was 0.35)
    
    # Pull the right side in so we aren't catching adjacent lanes
    right_bottom_x = int(width * 0.75) # Shifted left (was 0.85)
    
    # PINCH THE TOP INTO A TINY POINT
    # The vanishing point is very small, the top of the trapezoid must match it.
    left_top_x = int(width * 0.49)     # Almost perfectly center
    right_top_x = int(width * 0.51)    # Almost perfectly center
    
    src_points = np.float32([
        [left_bottom_x, bottom_y],     
        [right_bottom_x, bottom_y],    
        [right_top_x, horizon_y],      
        [left_top_x, horizon_y]        
    ])
    
    return src_points
def estimate_distance(image_path, vehicle_pixel_coord):
    # 1. Read the image to extract its native properties
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    height, width = img.shape[:2]
    
    # 2. Get image-driven automated source points
    src_points = get_automated_trapezoid(width, height)
    
    # 3. Define the destination BEV canvas size (Real-world aspect tracking)
    bev_width = 400
    bev_height = 600
    dst_points = np.float32([
        [0, bev_height],
        [bev_width, bev_height],
        [bev_width, 0],
        [0, 0]
    ])
    
    # 4. Calculate Homography
    H_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 5. Distance Mapping Scale (Based on your 1-meter camera mount setup)
    # Given a 1m camera height, an optimized horizon bounding box typically 
    # covers roughly 25 meters of visible roadway.
    max_visible_distance_meters = 25.0
    meters_per_pixel = max_visible_distance_meters / bev_height
    
    # 6. Transform the vehicle target point (must be bottom-center of its tires)
    u, v = vehicle_pixel_coord
    point_img = np.array([[[float(u), float(v)]]], dtype=np.float32)
    point_bev = cv2.perspectiveTransform(point_img, H_matrix)
    
    bev_x = point_bev[0][0][0]
    bev_y = point_bev[0][0][1]
    
    # 7. Compute real distance upwards from the bottom of the BEV frame
    pixel_distance = bev_height - bev_y
    distance_in_meters = pixel_distance * meters_per_pixel
    
    # Optional: Draw the trapezoid visually on a debug image to inspect accuracy
    debug_img = img.copy()
    pts = src_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)
    cv2.circle(debug_img, (u, v), 7, (0, 0, 255), -1)
    cv2.imwrite('automated_trapezoid_debug.jpg', debug_img)
    
    return distance_in_meters

if __name__ == "__main__":
    # Replace this point with the bottom-center tire pixel coordinates 
    # from your bounding box object detection pipeline output
    detected_vehicle_base = (640, 520) 
    
    try:
        dist = estimate_distance('frame_50.jpg', detected_vehicle_base)
        print(f"--- Calibration Success ---")
        print(f"Calculated vehicle distance: {dist:.2f} meters")
        print(f"Saved visualization diagnostic frame to: automated_trapezoid_debug.jpg")
    except Exception as e:
        print(f"Error: {e}")