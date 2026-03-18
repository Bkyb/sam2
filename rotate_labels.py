import os
from pathlib import Path

def rotate_yolo_labels_180(
    base_dir="export_results",
    target_subdir="valid",
    labels_dir_name="labels_6"
):
    """
    Rotates YOLO keypoint labels by 180 degrees for specified files.
    Applies x_new = 1 - x_old and y_new = 1 - y_old transformations.
    Keeps class_id as 0 and visibility tag as 2.
    Saves rotated labels to files with '_rotated.txt' suffix.
    """
    
    labels_path = Path(base_dir) / target_subdir / labels_dir_name
    
    if not labels_path.is_dir():
        print(f"Error: Labels directory not found at {labels_path}")
        return

    print(f"Processing labels in: {labels_path}")

    for label_file in labels_path.iterdir():
        if label_file.suffix == ".txt" and label_file.name.startswith("video_") and "_rotated" not in label_file.stem:
            print(f"Rotating {label_file.name}...")
            
            rotated_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    
                    if not parts:
                        continue

                    class_id = int(parts[0]) # Object index, keep as 0
                    
                    # Bounding box coordinates (normalized center_x, center_y, width, height)
                    bbox_center_x_old = parts[1]
                    bbox_center_y_old = parts[2]
                    bbox_width_old = parts[3]
                    bbox_height_old = parts[4]

                    bbox_center_x_new = 1.0 - bbox_center_x_old
                    bbox_center_y_new = 1.0 - bbox_center_y_old
                    # Width and height are unaffected by 180-degree rotation
                    bbox_width_new = bbox_width_old
                    bbox_height_new = bbox_height_old

                    new_line_parts = [str(class_id), 
                                      f"{bbox_center_x_new:.6f}", 
                                      f"{bbox_center_y_new:.6f}", 
                                      f"{bbox_width_new:.6f}", 
                                      f"{bbox_height_new:.6f}"]
                    
                    # Keypoint coordinates (normalized x, y, visibility)
                    for i in range(5, len(parts), 3):
                        if i + 2 < len(parts):
                            kp_x_old = parts[i]
                            kp_y_old = parts[i+1]
                            kp_v = int(parts[i+2]) # Visibility tag, keep as 2

                            kp_x_new = 1.0 - kp_x_old
                            kp_y_new = 1.0 - kp_y_old
                            
                            new_line_parts.extend([f"{kp_x_new:.6f}", f"{kp_y_new:.6f}", str(kp_v)])
            
                    rotated_lines.append(" ".join(new_line_parts))
            
            if rotated_lines:
                rotated_file_name = f"{label_file.stem}_rotated.txt"
                rotated_file_path = labels_path / rotated_file_name
                with open(rotated_file_path, 'w') as f:
                    f.write("\n".join(rotated_lines))
                print(f"Saved rotated labels to {rotated_file_path.name}")
            else:
                print(f"No valid lines found in {label_file.name}, skipping rotation.")

if __name__ == "__main__":
    print("Starting label rotation script...")
    rotate_yolo_labels_180()
    print("Label rotation script finished.")
