"""
Tomato Disease AI - YOLO26 Training Pipeline
Trains YOLO26 on tomato leaf disease dataset using transfer learning with full hyperparameter control.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

from src.utils.config import load_config, resolve_model_path, get_project_root
from src.utils.logger import setup_logger
from src.data.validate_dataset import validate_master_dataset

logger = setup_logger("train_yolo26")


def parse_args():
    cfg = load_config()
    train_cfg = cfg.get("training", {})
    infer_cfg = cfg.get("inference", {})

    parser = argparse.ArgumentParser(description="Train YOLO26 on Tomato Leaf Disease Dataset")
    parser.add_argument("--model", type=str, default="", help="Base model weights (e.g. yolo26n.pt, models/pretrained/yolo26n.pt)")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "dataset" / "data.yaml"), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 15), help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=train_cfg.get("batch", 8), help="Batch size")
    parser.add_argument("--imgsz", type=int, default=train_cfg.get("imgsz", 640), help="Image size")
    parser.add_argument("--patience", type=int, default=train_cfg.get("patience", 10), help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', or auto-detect")
    parser.add_argument("--workers", type=int, default=train_cfg.get("workers", 2), help="Worker threads")
    parser.add_argument("--lr", type=float, default=train_cfg.get("learning_rate", 0.001), help="Initial learning rate")
    parser.add_argument("--two-stage", action="store_true", default=train_cfg.get("two_stage_finetune", False), help="Enable two-stage fine-tuning")
    parser.add_argument("--name", type=str, default="tomato_yolo26_run", help="Experiment name")
    return parser.parse_args()


def run_training():
    args = parse_args()
    cfg = load_config()

    print("\n" + "=" * 75)
    print(" TOMATO LEAF DISEASE AI - YOLO26 TRAINING PIPELINE")
    print("=" * 75)

    # 1. Device Resolution
    if args.device:
        device = args.device
    else:
        cfg_dev = cfg.get("training", {}).get("device", "auto")
        if cfg_dev == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg_dev

    if torch.cuda.is_available() and device != "cpu":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: CUDA\nGPU: {gpu_name}")
    else:
        print("Device: CPU")

    # 2. Mandatory Pre-flight Dataset Validation
    ds_dir = PROJECT_ROOT / "dataset"
    dy_path = Path(args.data)
    print(f"\n[*] Executing Pre-flight Dataset Verification on: {dy_path.relative_to(PROJECT_ROOT)}...")
    passed, report, errors = validate_master_dataset(ds_dir, dy_path)
    
    if not passed:
        print("\n" + "!" * 75)
        print("DATASET VALIDATION FAILED - TRAINING ABORTED")
        print("!" * 75)
        for e in errors:
            print(f"  [-] {e}")
        print("\nPlease fix the above dataset errors before training.")
        sys.exit(1)
    else:
        print(f"[+] Dataset pre-flight check passed: {report['total_images']} verified images across {report['num_classes']} classes.")

    # 3. Model Weight Resolution
    base_model = args.model if args.model else str(resolve_model_path())
    print(f"\n[*] Resolving Base YOLO26 Weights: {base_model}")
    model = YOLO(base_model)

    runs_project = PROJECT_ROOT / "results" / "training"
    runs_project.mkdir(parents=True, exist_ok=True)

    aug_cfg = cfg.get("augmentation", {})

    # 4. Training Execution
    if args.two_stage:
        print("\n[*] Commencing Stage 1: Fine-tuning detection head with frozen backbone...")
        model.train(
            data=str(dy_path),
            epochs=max(5, args.epochs // 2),
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            patience=args.patience,
            lr0=args.lr,
            freeze=10,  # Freeze initial backbone layers
            project=str(runs_project),
            name=f"{args.name}_stage1",
            exist_ok=True,
            plots=True,
            verbose=True,
            **aug_cfg
        )
        print("\n[*] Commencing Stage 2: Unfreezing full network for comprehensive fine-tuning...")
        stage1_best = runs_project / f"{args.name}_stage1" / "weights" / "best.pt"
        if stage1_best.exists():
            model = YOLO(str(stage1_best))
        
        train_results = model.train(
            data=str(dy_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            patience=args.patience,
            lr0=args.lr * 0.5,
            freeze=None,  # Full unfreeze
            project=str(runs_project),
            name=args.name,
            exist_ok=True,
            plots=True,
            verbose=True,
            **aug_cfg
        )
    else:
        print(f"\n[*] Launching Single-Stage YOLO26 Fine-tuning ({args.epochs} epochs)...")
        train_results = model.train(
            data=str(dy_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            patience=args.patience,
            lr0=args.lr,
            project=str(runs_project),
            name=args.name,
            exist_ok=True,
            plots=True,
            verbose=True,
            **aug_cfg
        )

    # 5. Export Optimal Weights
    run_dir = runs_project / args.name
    trained_dir = PROJECT_ROOT / "models" / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)
    
    src_best = run_dir / "weights" / "best.pt"
    src_last = run_dir / "weights" / "last.pt"

    dest_best = trained_dir / "best.pt"
    dest_last = trained_dir / "last.pt"

    if src_best.exists():
        shutil.copy2(src_best, dest_best)
        print(f"\n[+] Exported Optimal Weights -> {dest_best.relative_to(PROJECT_ROOT)}")

    if src_last.exists():
        shutil.copy2(src_last, dest_last)
        print(f"[+] Exported Last Epoch Weights -> {dest_last.relative_to(PROJECT_ROOT)}")

    # Copy plots to results/graphs/
    graphs_dir = PROJECT_ROOT / "results" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    for plot_file in run_dir.glob("*.png"):
        shutil.copy2(plot_file, graphs_dir / plot_file.name)

    print("\n" + "=" * 75)
    print(" YOLO26 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    return dest_best


if __name__ == "__main__":
    run_training()
