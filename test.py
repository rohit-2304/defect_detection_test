import os
import sys
import json
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("      YOLO11 Nano Testing & Evaluation Script")
    print("=" * 60)
    
    # Paths
    best_weights_path = os.path.abspath("output/yolo11n/train_results/weights/best.pt")
    data_yaml_path = os.path.abspath("data_local.yaml")
    output_dir = os.path.abspath("output/yolo11n")
    test_images_dir = os.path.abspath("data/test/images")
    
    print(f"Best Weights Path: {best_weights_path}")
    print(f"Dataset Config: {data_yaml_path}")
    print(f"Test Images: {test_images_dir}")
    print("-" * 60)
    
    # Verify weights exist
    if not os.path.exists(best_weights_path):
        print(f"ERROR: Trained model weights not found at {best_weights_path}.")
        print("Please run train.py first to train the model and generate weights.")
        sys.exit(1)
        
    # 1. Load trained model
    print("Loading best trained model...")
    try:
        model = YOLO(best_weights_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    print("-" * 60)
    print("Running evaluation on 'test' split...")
    
    # 2. Run evaluation
    try:
        metrics = model.val(
            data=data_yaml_path,
            split="test",
            device=0,
            plots=True,
            project=output_dir,
            name="test_evaluation"
        )
        
        # Extract performance metrics
        map50 = float(metrics.box.map50)
        map95 = float(metrics.box.map)
        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        fitness = float(metrics.fitness)
        
        print("\nTest Evaluation Summary:")
        print(f"  - mean Precision (mP): {mp:.4f}")
        print(f"  - mean Recall (mR):    {mr:.4f}")
        print(f"  - mAP50:               {map50:.4f}")
        print(f"  - mAP50-95:            {map95:.4f}")
        print(f"  - Fitness Score:       {fitness:.4f}")
        
        # Save metrics to JSON
        metrics_dict = {
            "model_version": "YOLO11 Nano (yolo11n)",
            "metrics": {
                "mean_precision": mp,
                "mean_recall": mr,
                "mAP50": map50,
                "mAP50_95": map95,
                "fitness_score": fitness
            },
            "class_indices_to_names": model.names
        }
        
        metrics_json_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"\nSaved test metrics JSON to: {metrics_json_path}")
        
    except Exception as e:
        print(f"Warning: Evaluation failed or test labels were incomplete: {e}")
        print("We will still proceed with generating predictions and visual annotations.")
        metrics_dict = {"status": "Evaluation failed/skipped", "error": str(e)}

    print("-" * 60)
    print("Generating predictions and annotated images for test set...")
    
    # 3. Perform prediction and save annotated images
    try:
        results = model.predict(
            source=test_images_dir,
            save=True,
            imgsz=512,
            conf=0.25,  # Confidence threshold for visualization
            project=output_dir,
            name="test_annotated",
            exist_ok=True,
            device=0
        )
        
        # List generated images
        annotated_dir = os.path.join(output_dir, "test_annotated")
        print("\nPrediction visual results generated:")
        if os.path.exists(annotated_dir):
            files = os.listdir(annotated_dir)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"  - {os.path.join(annotated_dir, file)}")
        else:
            print("  Note: Predictions saved by YOLO CLI. Check output/yolo11n/test_annotated/ folder.")
            
        print("=" * 60)
        print("Inference completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
