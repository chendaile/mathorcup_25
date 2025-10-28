import argparse
import logging
from datetime import datetime

from ultralytics import YOLO

logging.basicConfig(
    filemode="w",
    filename="training-log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model with custom parameters."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolo11n.pt",
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--data", type=str, default="data.yaml", help="Path to the dataset YAML file."
    )
    parser.add_argument(
        "--batch", type=float, default=0.9, help="Batch size percentage for training."
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        type=int,
        default=640,
        help="Image size for training (e.g., 640, 1280).",
    )
    parser.add_argument(
        "--cache",
        type=bool,
        default=True,
        help="Whether to cache images for faster training.",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        required=True,
        help="Training time limit (in hours).",
    )
    time_str = str(datetime.now().strftime("%Y%m%d_hour%H_min%M"))
    parser.add_argument(
        "--name",
        type=str,
        default=time_str,
        help="Name to save training results with timestamp.",
    )
    parser.add_argument(
        "--project",
        "-P",
        type=str,
        required=True,
        help="Project to save training results with timestamp.",
    )
    parser.add_argument(
        "--resume",
        "-R",
        type=str,
        default=False,
        help="Whether continue to train from un_finished model",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "obb", "segment", "classify", "pose"],
        help="Task type: detect, obb, segment, classify, or pose.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Epochs to wait for no observable improvement for early stopping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run on, e.g., 0 or 0,1,2,3 or cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info("Training YOLO model with the following parameters:")
    logging.info(f"  Model:    {args.model}")
    logging.info(f"  Data:     {args.data}")
    logging.info(f"  Task:     {args.task}")
    logging.info(f"  Batch:    {args.batch}")
    logging.info(f"  Imgsz:    {args.imgsz}")
    logging.info(f"  Cache:    {args.cache}")
    logging.info(f"  Time:     {args.time}")
    logging.info(f"  Epochs:   {args.epochs}")
    logging.info(f"  Patience: {args.patience}")
    logging.info(f"  Device:   {args.device}")
    logging.info(f"  Name:     {args.name}")
    logging.info(f"  Project:  {args.project}")
    logging.info(f"  Resume:   {args.resume}")

    # 加载模型
    model = YOLO(model=args.model, task=args.task, verbose=False)
    
    # 训练模型
    model.train(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        cache=args.cache,
        time=args.time,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
        project=f"output-bin/{args.project}",
    )


if __name__ == "__main__":
    main()