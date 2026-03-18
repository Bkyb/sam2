import cv2
import numpy as np
import os
from pathlib import Path

# --- Constants ---
BASE_DIR = "export_results"
SUBDIRS = ["test", "train", "valid"]
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# --- Helper Functions (adapted from sam_app.py) ---

def get_image_dimensions(image_path):
    """Reads an image to get its width and height."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None, None
        h, w, _ = img.shape
        return w, h
    except Exception as e:
        print(f"Error reading image dimensions for {image_path}: {e}")
        return None, None

def load_yolo_labels(label_path, img_w, img_h):
    """
    Parses a labels_0/*.txt file and returns a list of dictionaries.
    Each dict contains class_id, bbox (normalized), and keypoints (pixel and normalized).
    """
    objects_data = []
    if not label_path.exists():
        return objects_data

    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            
            # YOLO bbox format: class_id center_x center_y width height
            bbox_norm = parts[1:5] 
            
            # Keypoints start after bbox. Each keypoint is x y visibility.
            keypoints_norm = []
            keypoints_pixel = []
            for i in range(5, len(parts), 3):
                if i + 2 < len(parts): # Ensure x, y, v are present
                    x_norm, y_norm, v = parts[i], parts[i+1], int(parts[i+2])
                    keypoints_norm.append([x_norm, y_norm, v])
                    keypoints_pixel.append([int(x_norm * img_w), int(y_norm * img_h), v])
            
            objects_data.append({
                'class_id': class_id,
                'bbox_norm': bbox_norm,
                'keypoints_norm': keypoints_norm,
                'keypoints_pixel': keypoints_pixel
            })
    return objects_data

def load_contour_data(contour_path, img_w, img_h):
    """
    Parses a contours/*.txt file and returns a list of polygons (pixel coordinates).
    Assumes each line is a single polygon with normalized x, y coordinates.
    """
    polygons_data = []
    if not contour_path.exists():
        return polygons_data

    with open(contour_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            
            # Each polygon is a flat list of normalized x, y coordinates
            polygon_norm = []
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    polygon_norm.append([parts[i], parts[i+1]])
            
            if polygon_norm:
                # Convert to pixel coordinates and reshape for OpenCV
                polygon_pixel = np.array([[int(p[0] * img_w), int(p[1] * img_h)] for p in polygon_norm], dtype=np.int32)
                polygons_data.append(polygon_pixel.reshape(-1, 1, 2))
    return polygons_data

def refine_polygon(original_polygon):
    """
    Refines a polygon by adding intermediate points, similar to sam_app.py's save_polygon.
    original_polygon: numpy array of shape (N, 1, 2) in pixel coordinates.
    Returns: refined polygon as numpy array of shape (M, 1, 2).
    """
    if original_polygon is None or len(original_polygon) < 2:
        return original_polygon

    refined_points_list = []
    points = original_polygon.reshape(-1, 2) # Flatten to (N, 2)
    num_points = len(points)

    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points] # Wrap around for the last segment

        refined_points_list.append(p1) # Add the current point

        distance = np.linalg.norm(p1 - p2)

        # Add intermediate points based on distance
        if distance > 4:
            # Add multiple intermediate points
            num_intermediate = int(distance // 2) - 1 # Number of points to add between p1 and p2
            if num_intermediate > 0:
                for j in range(1, num_intermediate + 1):
                    new_point = p1 + (p2 - p1) * j / (num_intermediate + 1)
                    refined_points_list.append(new_point.astype(int))
        elif distance > 2:
            # Add one intermediate point
            new_point = p1 + (p2 - p1) * 0.5
            refined_points_list.append(new_point.astype(int))
            
    return np.array(refined_points_list, dtype=np.int32).reshape(-1, 1, 2)

def find_closest_point_on_polygon(poly, clicked_point):
    """Finds the closest point on a polygon to a given clicked point."""
    poly_points = poly.reshape(-1, 2)
    distances = np.linalg.norm(poly_points - clicked_point, axis=1)
    return tuple(poly_points[np.argmin(distances)])

def get_path_indices(n, s_idx, e_idx, clockwise=True):
    """Gets indices along the polygon path."""
    path, curr = [], s_idx
    while curr != e_idx:
        path.append(curr)
        curr = (curr + 1 if clockwise else curr - 1 + n) % n
    path.append(e_idx)
    return path

def calculate_path_length(poly_points, indices):
    """Calculates the length of a path along polygon points."""
    if len(indices) < 2: return 0
    return sum(np.linalg.norm(poly_points[indices[i]] - poly_points[indices[i+1]]) for i in range(len(indices)-1))

def extract_side_points(poly_points, indices, offset_distance, num):
    """Extracts 'num' side points along a path with an optional offset."""
    if len(indices) < 2: return []
    seg_lens, new_pts = [0], []
    for i in range(1, len(indices)):
        p1, p2 = poly_points[indices[i-1]], poly_points[indices[i]]
        seg_lens.append(seg_lens[-1] + np.linalg.norm(p2 - p1))
    total_len = seg_lens[-1]
    if total_len == 0: return []
    
    # Calculate target lengths for intermediate points
    targets = [total_len * i / (num + 1) for i in range(1, num + 1)]
    
    for pos in targets:
        for i in range(1, len(seg_lens)):
            if seg_lens[i-1] <= pos <= seg_lens[i]:
                seg_len = seg_lens[i] - seg_lens[i-1]
                r = (pos - seg_lens[i-1]) / seg_len if seg_len > 0 else 0
                p1, p2 = poly_points[indices[i-1]], poly_points[indices[i]]
                interp = p1 + r * (p2 - p1)
                
                # Calculate normal for offset (if offset_distance is not 0)
                tangent = p2 - p1
                normal = np.array([tangent[1], -tangent[0]], dtype=np.float32)
                norm = np.linalg.norm(normal) + 1e-8
                normal /= norm
                
                new_pts.append(tuple((interp + normal * offset_distance).astype(int)))
                break
    return new_pts

def calculate_surrounding_points(poly, start_pt, end_pt, num, offset_distance=0):
    """
    Calculates surrounding points along the polygon between start_pt and end_pt.
    Returns a list of pixel (x, y) tuples.
    """
    poly_points = poly.reshape(-1, 2)
    n = len(poly_points)
    
    # Find indices of start_pt and end_pt on the polygon
    s_idx = np.argmin(np.linalg.norm(poly_points - start_pt, axis=1))
    e_idx = np.argmin(np.linalg.norm(poly_points - end_pt, axis=1))
    
    # Get paths in both clockwise and counter-clockwise directions
    path_cw = get_path_indices(n, s_idx, e_idx, True)
    path_ccw = get_path_indices(n, s_idx, e_idx, False)
    
    # Calculate lengths of both paths
    len_cw = calculate_path_length(poly_points, path_cw)
    len_ccw = calculate_path_length(poly_points, path_ccw)
    
    # Determine the longer and shorter paths
    long_path, short_path = (path_cw, path_ccw) if len_cw >= len_ccw else (path_ccw, path_cw)
    
    # Extract side points from both paths
    long_pts = extract_side_points(poly_points, long_path, offset_distance, num=num)
    short_pts = extract_side_points(poly_points, short_path, offset_distance, num=num)
    
    # Order the points based on cross product to maintain consistency
    # This part is directly from sam_app.py and ensures consistent ordering of side points
    if num == 0: # If no intermediate points are requested, just return start/end
        return long_pts + short_pts[::-1] # No start_pt, end_pt here, they are added later
    
    # Check orientation using cross product
    vec_start2end = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
    if long_pts: # Only if long_pts is not empty
        v_start2p = (long_pts[0][0] - start_pt[0], long_pts[0][1] - start_pt[1])
        cross_product = vec_start2end[0] * v_start2p[1] - vec_start2end[1] * v_start2p[0]
        if cross_product > 0:
            return long_pts + short_pts[::-1]
        else:
            return short_pts + long_pts[::-1]
    else: # If long_pts is empty, just return short_pts
        return short_pts + long_pts[::-1]


def generate_yolo_keypoint_annotation(class_id, bbox_norm, keypoints_pixel, img_w, img_h):
    """
    Generates a single YOLO keypoint annotation string.
    keypoints_pixel: list of [x, y, v] in pixel coordinates.
    """
    bbox_str = " ".join(map(str, bbox_norm))
    
    keypoints_str_parts = []
    for kp in keypoints_pixel:
        x_norm = kp[0] / img_w
        y_norm = kp[1] / img_h
        visibility = kp[2] if len(kp) > 2 else 2 # Default visibility to 2 if not provided
        keypoints_str_parts.append(f"{x_norm:.6f} {y_norm:.6f} {visibility}")
    
    keypoints_str = " ".join(keypoints_str_parts)
    
    return f"{class_id} {bbox_str} {keypoints_str}"

# --- Main Processing Logic ---

def process_directory():
    for subdir in SUBDIRS:
        image_dir = Path(BASE_DIR) / subdir / "images"
        labels_0_dir = Path(BASE_DIR) / subdir / "labels_0"
        contours_dir = Path(BASE_DIR) / subdir / "contours"
        labels_6_dir = Path(BASE_DIR) / subdir / "labels_6"
        
        labels_6_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing directory: {image_dir}")

        image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        
        for image_path in image_files:
            image_name = image_path.stem
            
            # Skip _rotated.txt files in valid subdir
            if subdir == "valid" and "_rotated" in image_name:
                print(f"Skipping _rotated file in valid subdir: {image_name}")
                continue

            img_w, img_h = get_image_dimensions(image_path)
            if img_w is None or img_h is None:
                continue # Skip if image dimensions could not be read

            label_0_path = labels_0_dir / f"{image_name}.txt"
            contour_path = contours_dir / f"{image_name}.txt"
            
            yolo_labels = load_yolo_labels(label_0_path, img_w, img_h)
            contour_data = load_contour_data(contour_path, img_w, img_h)

            if not yolo_labels:
                print(f"Warning: No labels found in {label_0_path}. Skipping {image_name}.")
                continue
            if not contour_data:
                print(f"Warning: No contours found in {contour_path}. Skipping {image_name}.")
                continue

            if len(yolo_labels) != len(contour_data):
                print(f"Warning: Mismatch in number of objects ({len(yolo_labels)}) and contours ({len(contour_data)}) for {image_name}. Skipping.")
                continue
            
            output_annotations = []
            
            for i in range(len(yolo_labels)):
                obj_label = yolo_labels[i]
                obj_contour = contour_data[i]
                
                refined_polygon = refine_polygon(obj_contour)
                
                # Derive q_point, e_point, r_click_point
                # Fallback to polygon points/centroid if not enough keypoints in labels_0
                q_point = None
                e_point = None
                r_click_point = None

                if len(obj_label['keypoints_pixel']) >= 1:
                    q_point = tuple(obj_label['keypoints_pixel'][0][:2])
                if len(obj_label['keypoints_pixel']) >= 2:
                    e_point = tuple(obj_label['keypoints_pixel'][1][:2])
                if len(obj_label['keypoints_pixel']) >= 3:
                    r_click_point = tuple(obj_label['keypoints_pixel'][2][:2])
                
                # Fallback if keypoints are missing
                if q_point is None and len(refined_polygon) > 0:
                    q_point = tuple(refined_polygon[0][0])
                if e_point is None and len(refined_polygon) > 1:
                    e_point = tuple(refined_polygon[1][0])
                elif e_point is None and len(refined_polygon) > 0: # If only one point, use it for both q and e
                    e_point = tuple(refined_polygon[0][0])

                if r_click_point is None and len(refined_polygon) > 0:
                    # Calculate centroid of the refined polygon
                    M = cv2.moments(refined_polygon)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        r_click_point = (cx, cy)
                    else: # Fallback if moments are zero (e.g., single point polygon)
                        r_click_point = tuple(refined_polygon[0][0])
                
                if q_point is None or e_point is None or r_click_point is None:
                    print(f"Warning: Could not determine q, e, or r points for object in {image_name}. Skipping object.")
                    continue

                # Calculate surrounding points (6 pairs = 12 points)
                # The num parameter in calculate_surrounding_points refers to the number of intermediate points *per side*
                surrounding_points_raw = calculate_surrounding_points(refined_polygon, q_point, e_point, num=6)
                
                # Combine q, e, r, and the 12 surrounding points
                # The visibility for q, e, r points is taken from labels_0 if available, otherwise default to 2
                # For surrounding points, default visibility to 2
                
                final_keypoints_pixel = []
                
                # Add q_point with its original visibility if available, else 2
                q_v = obj_label['keypoints_pixel'][0][2] if len(obj_label['keypoints_pixel']) >= 1 else 2
                final_keypoints_pixel.append([q_point[0], q_point[1], q_v])

                # Add e_point with its original visibility if available, else 2
                e_v = obj_label['keypoints_pixel'][1][2] if len(obj_label['keypoints_pixel']) >= 2 else 2
                final_keypoints_pixel.append([e_point[0], e_point[1], e_v])

                # Add r_click_point with its original visibility if available, else 2
                r_v = obj_label['keypoints_pixel'][2][2] if len(obj_label['keypoints_pixel']) >= 3 else 2
                final_keypoints_pixel.append([r_click_point[0], r_click_point[1], r_v])

                # Add the 12 surrounding points with default visibility 2
                for sp in surrounding_points_raw:
                    final_keypoints_pixel.append([sp[0], sp[1], 2])
                
                annotation_string = generate_yolo_keypoint_annotation(
                    obj_label['class_id'], 
                    obj_label['bbox_norm'], 
                    final_keypoints_pixel, 
                    img_w, img_h
                )
                output_annotations.append(annotation_string)
            
            if output_annotations:
                output_label_path = labels_6_dir / f"{image_name}.txt"
                with open(output_label_path, "w") as f:
                    f.write("\n".join(output_annotations))
                print(f"Generated {len(output_annotations)} annotations for {image_name}.txt in {labels_6_dir}")
            else:
                print(f"No annotations generated for {image_name}.txt")

if __name__ == "__main__":
    print("Starting labels_6 generation script...")
    process_directory()
    print("Labels_6 generation script finished.")