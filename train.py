import os
import sys
import time
import subprocess
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("      YOLO11 Nano Training Script - Defect Detection")
    print("=" * 60)
    
    # Run the data augmentation tool automatically before training
    print("Preparing and augmenting dataset...")
    try:
        subprocess.run([sys.executable, "augment.py"], check=True)
    except Exception as e:
        print(f"Error running augmentation: {e}")
        sys.exit(1)
        
    print("-" * 60)
    
    # Define paths
    data_yaml_path = os.path.abspath("data_local.yaml")
    project_dir = os.path.abspath("output/yolo11n")
    name = "train_results"
    
    print(f"Dataset Config: {data_yaml_path}")
    print(f"Output Project: {project_dir}")
    print(f"Run Directory Name: {name}")
    print("-" * 60)
    
    # 1. Initialize YOLO11 Nano model
    print("Loading YOLO11 Nano model (yolo11n.pt)...")
    try:
        model = YOLO("yolo11n.pt")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    print("-" * 60)
    print("Starting training process...")
    start_time = time.time()
    
    # 2. Train the model
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=100,             # Increased from 50 to 100 for augmented dataset
            imgsz=512,
            batch=4,
            device=0,               # Use CUDA GPU
            project=project_dir,
            name=name,
            exist_ok=True,
            plots=True,
            workers=2,
            verbose=True,
            
            # --- YOLO Online Data Augmentation Hyperparameters ---
            fliplr=0.5,             # Left-right flip probability
            flipud=0.5,             # Up-down flip probability
            mosaic=1.0,             # Mosaic augmentation probability
            mixup=0.15,             # Mixup augmentation probability
            hsv_h=0.015,            # HSV-Hue augmentation
            hsv_s=0.7,              # HSV-Saturation augmentation
            hsv_v=0.4               # HSV-Value augmentation
        )
        duration = time.time() - start_time
        print("-" * 60)
        print("Training completed successfully!")
        print(f"Total training time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Paths to weights
        weights_dir = os.path.join(project_dir, name, "weights")
        best_weights = os.path.join(weights_dir, "best.pt")
        last_weights = os.path.join(weights_dir, "last.pt")
        
        print(f"Best weights saved to: {best_weights}")
        print(f"Last weights saved to: {last_weights}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
