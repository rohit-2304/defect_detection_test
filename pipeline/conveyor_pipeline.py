import os
import sys
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 Conveyor Belt Defect Detection Pipeline")
    parser.add_argument(
        "--input", 
        type=str, 
        default="video_data/IMG_3252.mp4", 
        help="Path to input video file or webcam index"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output/yolo11n/conveyor_annotated.mp4", 
        help="Path to save annotated output video"
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default="output/yolo11n/train_results/weights/best.pt", 
        help="Path to trained YOLO model weights"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.15, 
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Inference image size (must match training resolution)"
    )
    parser.add_argument(
        "--target-height", 
        type=int, 
        default=960, 
        help="Target height for resized output video"
    )
    parser.add_argument(
        "--max-inference-dim",
        type=int,
        default=1024,
        help="Maximum dimension for inference frame (speeds up processing)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("      YOLO11 Conveyor Belt QA Pipeline")
    print("=" * 60)
    print(f"Input Video: {args.input}")
    print(f"Output Video: {args.output}")
    print(f"Weights Path: {args.weights}")
    print(f"Confidence Threshold: {args.conf}")
    print("-" * 60)
    
    # 1. Load model
    if not os.path.exists(args.weights):
        print(f"ERROR: Model weights not found at {args.weights}")
        print("Please train the model first or verify the path.")
        sys.exit(1)
        
    print("Loading YOLO model...")
    model = YOLO(args.weights)
    print("Model loaded successfully.")
    
    # 2. Open input video
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        if not os.path.exists(args.input):
            print(f"ERROR: Input video not found at {args.input}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.input)
        
    if not cap.isOpened():
        print(f"ERROR: Could not open video source {args.input}")
        sys.exit(1)
        
    # Get video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0
        
    print(f"Original Video: {orig_w}x{orig_h} @ {orig_fps:.2f} FPS ({total_frames} frames)")
    
    # Calculate target dimensions while preserving aspect ratio
    target_h = args.target_height
    target_w = int(orig_w * (target_h / orig_h))
    # Make sure width is even for video encoding
    if target_w % 2 != 0:
        target_w += 1
        
    print(f"Processing Target Size: {target_w}x{target_h}")
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 3. Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, orig_fps, (target_w, target_h))
    if not out.isOpened():
        print(f"ERROR: Could not open VideoWriter for {args.output}")
        sys.exit(1)
        
    # Class colors (BGR format)
    # 0: Dent, 1: Product, 2: Scratch, 3: Unclear surface, 4: chip off, 5: surface damage
    class_colors = {
        0: (0, 255, 255),    # Dent - Yellow
        1: (255, 100, 0),    # Product - Light Blue
        2: (0, 0, 255),      # Scratch - Red
        3: (255, 0, 255),    # Unclear surface - Purple/Magenta
        4: (0, 165, 255),    # chip off - Orange
        5: (0, 0, 128)       # surface damage - Maroon
    }
    
    # Cumulative stats
    defect_classes = {0, 2, 3, 4, 5}  # All classes except Product (1)
    unique_defect_ids = {c: set() for c in defect_classes}
    alerts_log = []
    
    # Fallback centroid tracking parameters for low-confidence defects
    active_fallback_tracks = {} # id -> (centroid_x, centroid_y, last_frame_seen, class_idx)
    next_fallback_id = 1000
    max_disappeared_frames = 15
    max_centroid_dist = 150 # in pixels
    
    frame_idx = 0
    start_time = time.time()
    
    print("Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        t_frame_start = time.time()
        
        # Resize frame to target resolution for visualization
        resized_frame = cv2.resize(frame, (target_w, target_h))
        
        # Scale down source frame if it exceeds max inference dimension (speeds up processing significantly)
        h, w, _ = frame.shape
        if max(h, w) > args.max_inference_dim:
            scale = args.max_inference_dim / max(h, w)
            inference_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            inference_frame = frame
            
        inf_h, inf_w, _ = inference_frame.shape
        
        # Run tracking on the scaled inference frame
        results = model.track(
            source=inference_frame,
            persist=True,
            conf=args.conf,
            imgsz=args.imgsz,
            device=0,
            verbose=False
        )
        
        # Current frame flags
        defect_detected_now = False
        current_defects = []
        
        # Clean up old active fallback tracks
        for fid in list(active_fallback_tracks.keys()):
            fx, fy, last_seen, fcls = active_fallback_tracks[fid]
            if frame_idx - last_seen > max_disappeared_frames:
                del active_fallback_tracks[fid]
        
        # Process results
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Draw bounding boxes and labels
            for box in boxes:
                # Get raw coordinates on full-res frame
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_idx = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Scale coordinates to target resolution
                rx1 = int(x1 * (target_w / inf_w))
                ry1 = int(y1 * (target_h / inf_h))
                rx2 = int(x2 * (target_w / inf_w))
                ry2 = int(y2 * (target_h / inf_h))
                
                # Get tracker ID if available
                track_id = int(box.id[0].item()) if box.id is not None else 0
                class_name = model.names[cls_idx]
                
                color = class_colors.get(cls_idx, (0, 255, 0))
                
                # Check if this is a defect class
                is_defect = cls_idx in defect_classes
                if is_defect:
                    defect_detected_now = True
                    
                    # Fallback tracking for lower confidence defects without a YOLO tracker ID
                    if track_id == 0:
                        cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
                        matched_id = None
                        best_dist = float('inf')
                        
                        for fid, (fx, fy, last_seen, fcls) in active_fallback_tracks.items():
                            if fcls == cls_idx:
                                dist = np.hypot(cx - fx, cy - fy)
                                if dist < best_dist and dist < max_centroid_dist:
                                    best_dist = dist
                                    matched_id = fid
                                    
                        if matched_id is not None:
                            active_fallback_tracks[matched_id] = (cx, cy, frame_idx, cls_idx)
                            track_id = matched_id
                        else:
                            # Register a new fallback track ID (starts at 1000)
                            track_id = next_fallback_id
                            active_fallback_tracks[track_id] = (cx, cy, frame_idx, cls_idx)
                            next_fallback_id += 1
                    
                    # Record unique tracker ID
                    if track_id > 0:
                        if track_id not in unique_defect_ids[cls_idx]:
                            unique_defect_ids[cls_idx].add(track_id)
                            log_msg = f"[{time.strftime('%H:%M:%S')}] New {class_name} detected! (ID: {track_id})"
                            alerts_log.append(log_msg)
                            # Keep log to last 5 entries
                            if len(alerts_log) > 5:
                                alerts_log.pop(0)
                    
                    current_defects.append(class_name)
                
                # Drawing bounding box on the resized frame
                # Thin bounding box for Product, thicker highlighted box for defects
                thickness = 2 if not is_defect else 4
                cv2.rectangle(resized_frame, (rx1, ry1), (rx2, ry2), color, thickness)
                
                # Draw neon glow effect for defects
                if is_defect:
                    # Draw a semi-transparent border around the box
                    overlay = resized_frame.copy()
                    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, 8)
                    cv2.addWeighted(overlay, 0.3, resized_frame, 0.7, 0, resized_frame)
                
                # Box label
                label_text = f"{class_name}"
                if track_id > 0:
                    label_text += f" #{track_id}"
                label_text += f" {conf:.2f}"
                
                # Draw label background
                (lbl_w, lbl_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(resized_frame, (rx1, ry1 - lbl_h - 10), (rx1 + lbl_w + 10, ry1), color, -1)
                # Label text
                cv2.putText(
                    resized_frame, 
                    label_text, 
                    (rx1 + 5, ry1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
        
        # Calculate Frame FPS
        t_frame_end = time.time()
        frame_fps = 1.0 / (t_frame_end - t_frame_start) if (t_frame_end - t_frame_start) > 0 else 30.0
        
        # --- Draw QA HUD Overlay ---
        # 1. Top Panel (Header)
        overlay = resized_frame.copy()
        cv2.rectangle(overlay, (0, 0), (target_w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, resized_frame, 0.15, 0, resized_frame)
        
        # Main Title
        cv2.putText(
            resized_frame, 
            "METALS QA CONVEYOR PIPELINE v1.0", 
            (20, 32), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        
        # Status Box (Pass / Warning Alert)
        if defect_detected_now:
            # Flashing warning (using frame index to toggle flash effect)
            is_flash = (frame_idx // 5) % 2 == 0
            status_color = (0, 0, 255) if is_flash else (0, 75, 255) # Bright red / dark red
            status_text = "[ALERT] DEFECT DETECTED!"
        else:
            status_color = (0, 200, 0) # Green
            status_text = "CONVEYOR STATUS: PASS"
            
        cv2.rectangle(resized_frame, (20, 45), (320, 70), status_color, -1)
        cv2.putText(
            resized_frame, 
            status_text, 
            (30, 63), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        
        # System status stats (top right)
        cv2.putText(
            resized_frame, 
            f"FPS: {frame_fps:.1f}", 
            (target_w - 220, 32), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (200, 200, 200), 
            1, 
            cv2.LINE_AA
        )
        cv2.putText(
            resized_frame, 
            f"Frame: {frame_idx}/{total_frames}", 
            (target_w - 220, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (200, 200, 200), 
            1, 
            cv2.LINE_AA
        )
        
        # 2. Left Overlay (Defect Counter Board)
        # Background block for stats
        overlay_stats = resized_frame.copy()
        cv2.rectangle(overlay_stats, (10, 90), (280, 280), (15, 15, 15), -1)
        cv2.addWeighted(overlay_stats, 0.75, resized_frame, 0.25, 0, resized_frame)
        
        # Title
        cv2.putText(
            resized_frame, 
            "QA DEFECT LOG COUNTER", 
            (20, 115), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 165, 255), 
            2, 
            cv2.LINE_AA
        )
        # Divider line
        cv2.line(resized_frame, (20, 125), (270, 125), (100, 100, 100), 1)
        
        # Defect Counts
        # Dent, Scratch, Unclear surface, chip off, surface damage
        counts_y = 150
        for cls_idx, class_name in model.names.items():
            if cls_idx in defect_classes:
                count = len(unique_defect_ids[cls_idx])
                color = class_colors.get(cls_idx, (255, 255, 255))
                cv2.putText(
                    resized_frame, 
                    f"{class_name.capitalize()}: {count}", 
                    (25, counts_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    1, 
                    cv2.LINE_AA
                )
                counts_y += 24
                
        # Total Unique Defects Tally
        total_defects = sum(len(unique_defect_ids[c]) for c in defect_classes)
        cv2.line(resized_frame, (20, counts_y - 8), (270, counts_y - 8), (100, 100, 100), 1)
        cv2.putText(
            resized_frame, 
            f"Total Defects: {total_defects}", 
            (20, counts_y + 12), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.55, 
            (0, 0, 255) if total_defects > 0 else (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        # 3. Bottom Alert Log HUD
        if alerts_log:
            overlay_log = resized_frame.copy()
            cv2.rectangle(overlay_log, (10, target_h - 130), (target_w - 10, target_h - 10), (10, 10, 10), -1)
            cv2.addWeighted(overlay_log, 0.8, resized_frame, 0.2, 0, resized_frame)
            
            cv2.putText(
                resized_frame, 
                "QA ALARM HISTORY LOGGER", 
                (20, target_h - 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            log_y = target_h - 85
            for log_entry in reversed(alerts_log):
                cv2.putText(
                    resized_frame, 
                    log_entry, 
                    (20, log_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (200, 200, 255), 
                    1, 
                    cv2.LINE_AA
                )
                log_y += 18
                
        # Write the frame
        out.write(resized_frame)
        
        # Progress logging
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
            
    # Cleanup
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print("-" * 60)
    print("Conveyor processing completed successfully!")
    print(f"Processed {frame_idx} frames in {total_time:.2f} seconds.")
    print(f"Average processing speed: {frame_idx / total_time:.2f} FPS")
    print(f"Annotated video saved: {args.output}")
    
    # Print final defect summary
    print("-" * 60)
    print("FINAL QUALITY CONTROL SUMMARY:")
    for cls_idx, class_name in model.names.items():
        if cls_idx in defect_classes:
            print(f"  - {class_name.capitalize()}: {len(unique_defect_ids[cls_idx])} unique item(s) found.")
    print("=" * 60)

if __name__ == "__main__":
    main()
