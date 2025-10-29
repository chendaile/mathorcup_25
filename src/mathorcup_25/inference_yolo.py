import argparse
import logging
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
import cv2
import os

logging.basicConfig(
    filemode="w",
    filename="inference-log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform inference on images using a YOLO model."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to the YOLO model file (e.g., model.pt).",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Path to input image or folder containing images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="inference_output",
        help="Output folder for prediction images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (0-1).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (0-1).",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        type=int,
        default=640,
        help="Image size for inference (e.g., 640, 1280).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run on, e.g., 0 or 0,1,2,3 or cpu",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "obb", "segment", "classify", "pose"],
        help="Task type: detect, obb, segment, classify, or pose.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save visualization results.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print detailed inference results.",
    )
    return parser.parse_args()


def get_image_files(source_path):
    """Get all image files from source path."""
    source = Path(source_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    image_files = []

    if source.is_file():
        if source.suffix.lower() in image_extensions:
            image_files.append(source)
    elif source.is_dir():
        for ext in image_extensions:
            image_files.extend(source.glob(f"*{ext}"))
            image_files.extend(source.glob(f"*{ext.upper()}"))
    else:
        logging.warning(f"Source path does not exist: {source_path}")

    return sorted(image_files)


def save_prediction_image(result, output_path):
    """Save prediction visualization."""
    # YOLO results contain the visualization in result.plot()
    plot_img = result.plot()
    cv2.imwrite(str(output_path), plot_img)


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("YOLO Inference Started")
    logging.info("=" * 60)
    logging.info(f"Model:        {args.model}")
    logging.info(f"Source:       {args.source}")
    logging.info(f"Output:       {args.output}")
    logging.info(f"Confidence:   {args.conf}")
    logging.info(f"IOU:          {args.iou}")
    logging.info(f"Image Size:   {args.imgsz}")
    logging.info(f"Device:       {args.device}")
    logging.info(f"Task:         {args.task}")
    logging.info("=" * 60)

    # Load model
    try:
        model = YOLO(model=args.model, task=args.task, verbose=args.verbose)
        logging.info(f"Model loaded successfully: {args.model}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

    # Get image files
    image_files = get_image_files(args.source)

    if not image_files:
        logging.warning(f"No image files found in {args.source}")
        print(f"No image files found in {args.source}")
        return

    logging.info(f"Found {len(image_files)} image(s) to process")
    print(f"Processing {len(image_files)} image(s)...")

    # Perform inference
    successful = 0
    failed = 0

    for idx, img_path in enumerate(image_files, 1):
        try:
            # Run inference
            results = model(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=args.verbose,
            )

            # Save prediction image
            result = results[0]
            output_filename = f"{img_path.stem}_pred{img_path.suffix}"
            output_path = output_dir / output_filename

            save_prediction_image(result, output_path)

            # Log statistics
            det_count = len(result.boxes) if result.boxes is not None else 0
            logging.info(f"[{idx}/{len(image_files)}] {img_path.name} - Detections: {det_count}, Saved: {output_path}")

            if args.verbose:
                print(f"[{idx}/{len(image_files)}] {img_path.name} - {det_count} detections")

            successful += 1

        except Exception as e:
            logging.error(f"[{idx}/{len(image_files)}] Failed to process {img_path.name}: {e}")
            print(f"[{idx}/{len(image_files)}] Error processing {img_path.name}: {e}")
            failed += 1

    # Summary
    logging.info("=" * 60)
    logging.info(f"Inference Complete")
    logging.info(f"  Successful: {successful}/{len(image_files)}")
    logging.info(f"  Failed:     {failed}/{len(image_files)}")
    logging.info(f"  Output:     {output_dir.absolute()}")
    logging.info("=" * 60)

    print("\n" + "=" * 60)
    print(f"Inference Complete!")
    print(f"  Successful: {successful}/{len(image_files)}")
    print(f"  Failed:     {failed}/{len(image_files)}")
    print(f"  Output:     {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
