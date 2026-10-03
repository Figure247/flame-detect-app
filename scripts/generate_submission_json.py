import argparse
import json
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import precision_recall_fscore_support
except ImportError:  # pragma: no cover - optional dependency
    precision_recall_fscore_support = None

from ultralytics import YOLO


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(directory: Path):
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def build_validation_truth(images_dir: Path, labels_dir: Path):
    true_labels = {}
    for image_path in list_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        # YOLO labels file is empty when no target is present.
        has_fire = 1 if label_path.exists() and label_path.stat().st_size > 0 else 0
        true_labels[image_path.name] = has_fire
    return true_labels


def predict_max_conf(model: YOLO, image_path: Path, conf_threshold: float = 0.001, imgsz: int = 640):
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf_threshold, verbose=False)
    max_conf = 0.0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        confs = boxes.conf
        if confs is not None and len(confs) > 0:
            max_conf = max(max_conf, float(confs.max().item()))
    return max_conf


def choose_threshold_by_validation(model: YOLO, validation_images_dir: Path, validation_labels_dir: Path, start: float = 0.05, stop: float = 0.95, step: float = 0.05):
    true_labels = build_validation_truth(validation_images_dir, validation_labels_dir)
    max_confs = {}
    for idx, image_path in enumerate(list_images(validation_images_dir), start=1):
        max_confs[image_path.name] = predict_max_conf(model, image_path)
        if idx % 50 == 0:
            print(f"已处理验证集图片 {idx}/{len(true_labels)}")

    if precision_recall_fscore_support is None:
        raise RuntimeError("需要安装 scikit-learn 以计算 Precision / Recall / F1")

    best_threshold = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    best_precision = 0.0

    for thresh in np.arange(start, stop, step):
        preds = {name: 1 if max_confs.get(name, 0.0) >= float(thresh) else 0 for name in true_labels}
        y_true = list(true_labels.values())
        y_pred = list(preds.values())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and recall > best_recall):
            best_f1 = float(f1)
            best_recall = float(recall)
            best_precision = float(precision)
            best_threshold = float(thresh)

    print(f"最佳图像级阈值: {best_threshold:.2f}")
    print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    return best_threshold


def generate_submission_from_test_set(model: YOLO, test_images_dir: Path, threshold: float, output_path: Path, imgsz: int = 640):
    submission = {}
    for image_path in list_images(test_images_dir):
        max_conf = predict_max_conf(model, image_path, conf_threshold=0.001, imgsz=imgsz)
        label = 1 if max_conf >= threshold else 0
        submission[image_path.name] = int(label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(submission, fp, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"已写入提交文件: {output_path}")
    print(f"有火图像数: {sum(submission.values())}")
    print(f"无火图像数: {len(submission) - sum(submission.values())}")
    return submission


def main():
    parser = argparse.ArgumentParser(description="生成比赛要求的图像级 0/1 提交 JSON")
    parser.add_argument("--model", required=True, help="YOLO 权重文件路径")
    parser.add_argument("--test-dir", required=True, help="测试集图片目录")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--validation-dir", default=None, help="验证集图片目录，用于搜索最佳图像级阈值")
    parser.add_argument("--labels-dir", default=None, help="YOLO 标签目录，与验证集图片对应")
    parser.add_argument("--threshold", type=float, default=None, help="直接使用固定阈值；若未给出则结合验证集搜索")
    parser.add_argument("--imgsz", type=int, default=640, help="模型推理图片尺寸")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.threshold is None:
        if not args.validation_dir or not args.labels_dir:
            raise ValueError("当未指定 --threshold 时，必须同时给出 --validation-dir 和 --labels-dir")
        threshold = choose_threshold_by_validation(
            model,
            Path(args.validation_dir),
            Path(args.labels_dir),
        )
    else:
        threshold = args.threshold

    submission = generate_submission_from_test_set(
        model,
        Path(args.test_dir),
        threshold=float(threshold),
        output_path=Path(args.output),
        imgsz=args.imgsz,
    )
    print(json.dumps(submission, ensure_ascii=False, indent=2, sort_keys=True)[:500])


if __name__ == "__main__":
    main()
