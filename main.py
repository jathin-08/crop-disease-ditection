"""
Tomato Disease AI - Master CLI Entrypoint
Unified interface for running dataset validation, YOLO26 training, validation,
independent evaluation, image/video inference, model export, and Streamlit app.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Tomato Leaf Disease AI - Advanced YOLO26 Diagnostic Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  validate         Run 25-point dataset pre-flight check
  train            Train YOLO26 model on tomato disease dataset
  validate_model   Run validation on validation split
  evaluate         Run benchmark on independent test dataset
  detect_image     Perform inference on image file or folder
  detect_video     Perform inference and tracking on video
  webcam           Start real-time webcam inference monitor
  export           Export best.pt checkpoint to deployment format
  app              Launch interactive Streamlit dashboard
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Target command to execute")

    # 1. validate
    sub_val = subparsers.add_parser("validate", help="Validate dataset integrity")

    # 2. train
    sub_train = subparsers.add_parser("train", help="Train YOLO26 model")
    sub_train.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    sub_train.add_argument("--batch", type=int, default=8, help="Batch size")
    sub_train.add_argument("--imgsz", type=int, default=640, help="Image size")
    sub_train.add_argument("--two-stage", action="store_true", help="Enable two-stage training")

    # 3. validate_model
    sub_vm = subparsers.add_parser("validate_model", help="Validate trained model")
    sub_vm.add_argument("--model", type=str, default="", help="Model weights path")
    sub_vm.add_argument("--split", type=str, default="val", help="Split to evaluate")

    # 4. evaluate
    sub_eval = subparsers.add_parser("evaluate", help="Evaluate independent test dataset")
    sub_eval.add_argument("--model", type=str, default="", help="Model weights path")
    sub_eval.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    sub_eval.add_argument("--iou", type=float, default=0.50, help="IoU threshold")

    # 5. detect_image
    sub_img = subparsers.add_parser("detect_image", help="Run image detection")
    sub_img.add_argument("--source", type=str, required=True, help="Image path or folder")
    sub_img.add_argument("--model", type=str, default="", help="Model weights path")
    sub_img.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")

    # 6. detect_video
    sub_vid = subparsers.add_parser("detect_video", help="Run video detection")
    sub_vid.add_argument("--source", type=str, default=str(PROJECT_ROOT / "videos" / "input" / "tomato_sample_video.mp4"), help="Input video path")
    sub_vid.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "predictions" / "videos" / "tomato_disease_result.mp4"), help="Output video path")
    sub_vid.add_argument("--model", type=str, default="", help="Model weights path")
    sub_vid.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    sub_vid.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    sub_vid.add_argument("--step", type=int, default=1, help="Frame step / sampling rate")
    sub_vid.add_argument("--labels", type=str, default="", help="Ground-truth labels directory")

    # 6b. video_evaluate
    sub_ve = subparsers.add_parser("video_evaluate", help="Run video detection and ground-truth evaluation")
    sub_ve.add_argument("--source", type=str, default=str(PROJECT_ROOT / "videos" / "input" / "tomato_sample_video.mp4"), help="Input video path")
    sub_ve.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "predictions" / "videos" / "tomato_disease_result.mp4"), help="Output video path")
    sub_ve.add_argument("--model", type=str, default="", help="Model weights path")
    sub_ve.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    sub_ve.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    sub_ve.add_argument("--step", type=int, default=1, help="Frame step / sampling rate")
    sub_ve.add_argument("--labels", type=str, default="", help="Ground-truth labels directory")

    # 7. webcam
    sub_cam = subparsers.add_parser("webcam", help="Start webcam monitoring")
    sub_cam.add_argument("--camera", type=int, default=0, help="Camera device index")

    # 8. export
    sub_exp = subparsers.add_parser("export", help="Export model")
    sub_exp.add_argument("--model", type=str, default="", help="Model weights path")
    sub_exp.add_argument("--format", type=str, default="onnx", help="Export format")

    # 9. app
    sub_app = subparsers.add_parser("app", help="Launch Streamlit Web App")

    # 10. iou_eval
    sub_iou = subparsers.add_parser("iou_eval", help="Run standalone IoU evaluation on dataset split")
    sub_iou.add_argument("--model", type=str, default="", help="Model weights path")
    sub_iou.add_argument("--split", type=str, default="val", help="Dataset split ('val' or 'test')")
    sub_iou.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "validate":
        from src.data.validate_dataset import main as val_main
        val_main()
    elif args.command == "train":
        from src.training.train import run_training
        run_training()
    elif args.command == "validate_model":
        from src.training.validate import run_validation
        run_validation(model_path=args.model, split=args.split)
    elif args.command == "evaluate":
        from src.training.evaluate import main as eval_main
        eval_main(args)
    elif args.command == "detect_image":
        from src.inference.image_detection import main as img_main
        img_main()
    elif args.command in ["detect_video", "video_evaluate"]:
        from src.inference.video_detection import process_video_stream, resolve_model_path
        from ultralytics import YOLO
        src_p = Path(args.source)
        if not src_p.exists():
            print(f"[x] Input video not found: {src_p}")
            sys.exit(1)
        model_p = args.model if args.model else str(resolve_model_path())
        print(f"[*] Loading YOLO Checkpoint: {model_p}")
        model = YOLO(model_p)
        lbl_p = Path(args.labels) if args.labels else None
        process_video_stream(
            source_video=src_p,
            output_video=Path(args.output),
            model=model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            frame_step=args.step,
            labels_dir=lbl_p
        )
    elif args.command == "webcam":
        from src.inference.webcam_detection import main as cam_main
        cam_main()
    elif args.command == "export":
        from src.deployment.export_model import main as exp_main
        exp_main()
    elif args.command == "app":
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
    elif args.command == "iou_eval":
        from ultralytics import YOLO
        from src.utils.config import resolve_model_path, get_classes
        from src.metrics.iou_metrics import evaluate_iou, export_iou_csv, generate_iou_bar_chart
        model_path = args.model if args.model else str(resolve_model_path())
        print(f"\n[*] Loading model for IoU Evaluation: {model_path}")
        model = YOLO(model_path)
        classes = get_classes()
        summary = evaluate_iou(
            model=model,
            dataset_dir=PROJECT_ROOT / "dataset",
            split=args.split,
            conf=args.conf,
            class_names=classes
        )
        print("\n" + "=" * 55)
        print(f" IoU EVALUATION REPORT - {args.split.upper()} SPLIT")
        print("=" * 55)
        print(f"Evaluated Images     : {summary['evaluated_images']}")
        print(f"Evaluated Detections : {summary['evaluated_detections']}")
        print(f"Mean IoU             : {summary['mean_iou']:.4f} ({summary['mean_iou']*100:.2f}%)")
        print(f"IoU >= 0.50 Threshold: {summary['iou_gte_50_pct']:.2f}% ({summary['iou_gte_50_count']}/{summary['evaluated_detections']})")
        print(f"IoU >= 0.75 Threshold: {summary['iou_gte_75_pct']:.2f}% ({summary['iou_gte_75_count']}/{summary['evaluated_detections']})")
        print(f"Min / Max IoU        : {summary['min_iou']:.4f} / {summary['max_iou']:.4f}")
        print("-" * 55)
        print("PER-CLASS IoU BREAKDOWN:")
        print(f"  {'Disease Name':<30} | {'Count':<6} | {'Mean IoU':<9} | {'IoU>=0.50':<9} | {'IoU>=0.75'}")
        print("  " + "-" * 72)
        for pc in summary.get("per_class_iou", []):
            print(f"  {pc['disease_name']:<30} | {pc['evaluated_detections']:<6} | {pc['mean_iou']:<9.4f} | {pc['iou_gte_50_pct']:<8.1f}% | {pc['iou_gte_75_pct']:.1f}%")
        print("=" * 55)
        out_csv = PROJECT_ROOT / "results" / f"iou_results_{args.split}.csv"
        export_iou_csv(summary.get("matches", []), out_csv)
        chart_p = PROJECT_ROOT / "results" / "graphs" / f"iou_per_class_{args.split}.png"
        generate_iou_bar_chart(summary.get("per_class_iou", []), chart_p)
        print(f"[+] IoU results exported to: {out_csv.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
