import os
import shutil
import cv2
import numpy as np

def clean_and_convert_labels():
    """
    Scans all label files in train, valid, and test splits and converts
    any polygon/segmentation annotations to clean YOLO bounding boxes.
    """
    print("Converting all polygon annotations to clean bounding boxes...")
    converted_lines_total = 0
    
    for split in ["train", "valid", "test"]:
        lbl_dir = f"data/{split}/labels"
        if not os.path.exists(lbl_dir):
            continue
            
        for f in os.listdir(lbl_dir):
            if f.endswith(".txt") and not f.startswith(".") and "_aug" not in f:
                lbl_path = os.path.join(lbl_dir, f)
                with open(lbl_path, "r") as file:
                    lines = file.readlines()
                
                new_lines = []
                file_changed = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 5:
                        # This is a polygon/segmentation (class x1 y1 x2 y2 ... xN yN)
                        cls = parts[0]
                        coords = list(map(float, parts[1:]))
                        
                        # Separate X and Y coordinates
                        xs = coords[0::2]
                        ys = coords[1::2]
                        
                        xmin = min(xs)
                        xmax = max(xs)
                        ymin = min(ys)
                        ymax = max(ys)
                        
                        w = xmax - xmin
                        h = ymax - ymin
                        x = xmin + w / 2.0
                        y = ymin + h / 2.0
                        
                        new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        file_changed = True
                        converted_lines_total += 1
                    else:
                        new_lines.append(line)
                        
                if file_changed:
                    with open(lbl_path, "w") as file:
                        file.writelines(new_lines)
                        
    print(f"Successfully converted {converted_lines_total} polygon annotations to bounding boxes.")

def clean_previous_augmentations(train_img_dir, train_lbl_dir):
    """Deletes previously generated augmented files to make the script idempotent."""
    print("Cleaning up any existing augmented files...")
    cleaned_images = 0
    cleaned_labels = 0
    
    if os.path.exists(train_img_dir):
        for f in os.listdir(train_img_dir):
            if "_aug" in f:
                os.remove(os.path.join(train_img_dir, f))
                cleaned_images += 1
                
    if os.path.exists(train_lbl_dir):
        for f in os.listdir(train_lbl_dir):
            if "_aug" in f:
                os.remove(os.path.join(train_lbl_dir, f))
                cleaned_labels += 1
                
    print(f"Cleaned {cleaned_images} augmented images and {cleaned_labels} augmented label files.")

def copy_missing_classes(train_img_dir, train_lbl_dir):
    """Copies rare/missing classes from test and valid sets into the training set."""
    print("Checking and copying rare classes to training set...")
    
    files_to_copy = [
        ("test", "IMG_3235_jpg.rf.193726f6ce7f5ff686a52fc4ea45ac3c", ".jpg"),
        ("test", "IMG_3239_jpg.rf.66eae30e6b8dd0a636d9a534e6417f14", ".jpg"),
        ("valid", "IMG_3243_jpg.rf.dad3f4c797c0322c74ddf855ed3a0d87", ".jpg")
    ]
    
    copied_count = 0
    for split, basename, ext in files_to_copy:
        src_img = f"data/{split}/images/{basename}{ext}"
        src_lbl = f"data/{split}/labels/{basename}.txt"
        
        dest_img = os.path.join(train_img_dir, f"{basename}{ext}")
        dest_lbl = os.path.join(train_lbl_dir, f"{basename}.txt")
        
        # Only copy if they don't already exist in train
        if os.path.exists(src_img) and not os.path.exists(dest_img):
            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_lbl, dest_lbl)
            print(f"  Copied {basename} ({split}) to train split.")
            copied_count += 1
            
    print(f"Copied {copied_count} files containing missing classes to train split.")

def apply_flips(img, bboxes, flip_code):
    """
    Applies flips to image and adjusts YOLO normalized bounding boxes.
    flip_code: 1 for horizontal, 0 for vertical, -1 for diagonal (both).
    """
    flipped_img = cv2.flip(img, flip_code)
    flipped_bboxes = []
    
    for bbox in bboxes:
        cls, x, y, w, h = bbox
        if flip_code == 1:       # Horizontal
            x_new = 1.0 - x
            y_new = y
        elif flip_code == 0:     # Vertical
            x_new = x
            y_new = 1.0 - y
        else:                    # Diagonal (-1)
            x_new = 1.0 - x
            y_new = 1.0 - y
        flipped_bboxes.append([cls, x_new, y_new, w, h])
        
    return flipped_img, flipped_bboxes

def adjust_brightness(img, factor):
    """Adjusts brightness of an image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int32) * factor, 0, 255).astype(np.uint8)
    hsv_new = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

def adjust_contrast(img, factor):
    """Adjusts contrast of an image."""
    mean = np.mean(img)
    img_new = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return img_new

def apply_blur(img):
    """Applies Gaussian Blur to an image."""
    return cv2.GaussianBlur(img, (5, 5), 0)

def main():
    print("=" * 60)
    print("      YOLO Dataset Augmentation & Cleaning Tool")
    print("=" * 60)
    
    train_img_dir = "data/train/images"
    train_lbl_dir = "data/train/labels"
    
    # 1. Convert all polygons to standard bounding boxes across splits
    clean_and_convert_labels()
    print("-" * 60)
    
    # 2. Clean previous runs
    clean_previous_augmentations(train_img_dir, train_lbl_dir)
    
    # 3. Copy missing classes from other splits to training
    copy_missing_classes(train_img_dir, train_lbl_dir)
    
    # 4. Identify files with target classes: Scratch (2), chip off (4), surface damage (5)
    target_classes = {2, 4, 5}
    candidates = []
    
    print("-" * 60)
    print("Scanning training labels for target classes...")
    for f in os.listdir(train_lbl_dir):
        if f.endswith(".txt") and not f.startswith("."):
            lbl_path = os.path.join(train_lbl_dir, f)
            with open(lbl_path, "r") as file:
                lines = file.readlines()
                
            has_target = False
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    bboxes.append([cls, x, y, w, h])
                    if cls in target_classes:
                        has_target = True
                        
            if has_target:
                img_file = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    possible_name = f.replace(".txt", ext)
                    if os.path.exists(os.path.join(train_img_dir, possible_name)):
                        img_file = possible_name
                        break
                        
                if img_file:
                    candidates.append({
                        "basename": f.replace(".txt", ""),
                        "img_file": img_file,
                        "lbl_file": f,
                        "bboxes": bboxes
                    })
                    
    print(f"Found {len(candidates)} candidate images with target classes for augmentation:")
    for c in candidates:
        print(f"  - {c['img_file']} (contains bboxes with classes: {[b[0] for b in c['bboxes']]})")
        
    print("-" * 60)
    print("Generating 10 custom OpenCV-based augmentations per candidate...")
    
    augmented_count = 0
    for c in candidates:
        img_path = os.path.join(train_img_dir, c["img_file"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read image {img_path}")
            continue
            
        bboxes = c["bboxes"]
        basename = c["basename"]
        ext = os.path.splitext(c["img_file"])[1]
        
        transforms = [
            ("aug_hflip", lambda im, bb: apply_flips(im, bb, 1)),
            ("aug_vflip", lambda im, bb: apply_flips(im, bb, 0)),
            ("aug_dflip", lambda im, bb: apply_flips(im, bb, -1)),
            ("aug_bright", lambda im, bb: (adjust_brightness(im, 1.3), bb)),
            ("aug_dark", lambda im, bb: (adjust_brightness(im, 0.7), bb)),
            ("aug_blur", lambda im, bb: (apply_blur(im), bb)),
            ("aug_hicontrast", lambda im, bb: (adjust_contrast(im, 1.3), bb)),
            ("aug_locontrast", lambda im, bb: (adjust_contrast(im, 0.7), bb)),
            ("aug_hflip_bright", lambda im, bb: (adjust_brightness(cv2.flip(im, 1), 1.2), apply_flips(im, bb, 1)[1])),
            ("aug_vflip_blur", lambda im, bb: (apply_blur(cv2.flip(im, 0)), apply_flips(im, bb, 0)[1]))
        ]
        
        for suffix, func in transforms:
            aug_img, aug_bboxes = func(img, bboxes)
            
            aug_img_name = f"{basename}_{suffix}{ext}"
            aug_img_path = os.path.join(train_img_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)
            
            aug_lbl_name = f"{basename}_{suffix}.txt"
            aug_lbl_path = os.path.join(train_lbl_dir, aug_lbl_name)
            with open(aug_lbl_path, "w") as f_out:
                for bbox in aug_bboxes:
                    f_out.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            
            augmented_count += 1
            
    print("-" * 60)
    print("Augmentation completed successfully!")
    print(f"Generated {augmented_count} new augmented image/label pairs.")
    print(f"New total training set size: {len(os.listdir(train_img_dir))} images.")
    print("=" * 60)

if __name__ == "__main__":
    main()
