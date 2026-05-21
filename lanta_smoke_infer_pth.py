#!/usr/bin/env python3
"""Run PyTorch checkpoint inference on an ImageFolder-style sample set.
ตอนนี้โมเดล shufflenet_v2_x0_5 ยังมี output class ชื่อ Unknown อยู่ แต่เราจะตัด Unknown แบบ post-process ตอน inference แทนการ retrain

วิธีทำ:
หลังจากได้ logits/probabilities จากโมเดล ให้ set ค่า class Unknown เป็น -inf หรือ 0 probability ก่อนเลือก top-1 prediction

เช่น:
Unknown index = 22
logits[:, 22] = -inf
แล้วค่อย argmax ใหม่

ผลคือโมเดลจะไม่มีทาง predict เป็น Unknown ตอนใช้งานจริง แต่ architecture/checkpoint เดิมยังใช้ได้เหมือนเดิม ไม่ต้อง train ใหม่ ไม่ต้องเปลี่ยนหัว classifier

ข้อดี:
ทำเร็วมาก
ไม่ต้อง retrain
ไม่ต้องแก้ checkpoint
แก้เฉพาะ inference/export wrapper ได้เลย

ข้อควรระวัง:
ถ้ารูปจริง ๆ ควรเป็น Unknown โมเดลจะถูกบังคับให้เลือก brand อื่นแทน
ดังนั้นควรใช้เฉพาะกรณีที่ deployment ต้องการให้ตอบเป็น brand เสมอ
ถ้าต้องการ reject unknown จริง ๆ ควรใช้ confidence threshold เพิ่มอีกชั้น

จาก test set หลังตัด true_label == Unknown ออก:
Original accuracy = 0.823583
Original macro F1 = 0.583476
Post-process ignore Unknown accuracy = 0.849177
Post-process ignore Unknown macro F1 = 0.585595

สรุป:
post-process แบบ ignore Unknown head ช่วยให้ accuracy ดีขึ้นจาก 82.36% เป็น 84.92% โดยไม่ต้อง retrain

"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DISALLOWED_PREDICTION_LABELS = {"Unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small PTH smoke inference test."
    )
    parser.add_argument("--bundle-dir", type=Path, default=Path("lanta_pth_bundle"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Infer only this checkpoint. Accepts an absolute path or a path relative to "
            "--bundle-dir, e.g. models_data3_track_stratified_fp16_f1/shufflenet_v2_x0_5_best.pth."
        ),
    )
    parser.add_argument("--sample-dir", type=Path, default=Path("smoke_test_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_test_outputs"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def replace_classifier(model: nn.Module, num_classes: int, dropout_p: float) -> None:
    if not hasattr(model, "classifier") or not isinstance(
        model.classifier, nn.Sequential
    ):
        raise TypeError(f"Unsupported MobileNet classifier: {type(model)}")
    original_last_layer = model.classifier[-1]
    if not isinstance(original_last_layer, nn.Linear):
        raise TypeError(
            f"Expected final classifier layer to be nn.Linear, got {original_last_layer}"
        )
    for layer in model.classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout_p
    if not any(isinstance(layer, nn.Dropout) for layer in model.classifier):
        model.classifier = nn.Sequential(
            *list(model.classifier[:-1]),
            nn.Dropout(p=dropout_p),
            model.classifier[-1],
        )
    model.classifier[-1] = nn.Linear(original_last_layer.in_features, num_classes)


def infer_model_name(checkpoint_path: Path, checkpoint: dict) -> str:
    model_name = checkpoint.get("model_name", "")
    if model_name:
        return str(model_name)
    path_text = checkpoint_path.as_posix()
    for candidate in (
        "mobilenet_v2_tiny_width_035",
        "mobilenet_v2_full",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x0_5",
    ):
        if candidate in path_text:
            return candidate
    raise ValueError(f"Cannot infer model type for {checkpoint_path}")


def build_model(model_name: str, num_classes: int, dropout_p: float) -> nn.Module:
    if model_name == "mobilenet_v2_full":
        model = models.mobilenet_v2(weights=None)
        replace_classifier(model, num_classes, dropout_p)
        return model
    if model_name == "mobilenet_v2_tiny_width_035":
        model = models.mobilenet_v2(weights=None, width_mult=0.35)
        replace_classifier(model, num_classes, dropout_p)
        return model
    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        replace_classifier(model, num_classes, dropout_p)
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        replace_classifier(model, num_classes, dropout_p)
        return model
    if model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_name={model_name}")


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((224, 224))
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(224, 224, 3).permute(2, 0, 1).float().div(255.0)
    return (data - IMAGENET_MEAN) / IMAGENET_STD


def list_samples(sample_dir: Path) -> list[tuple[Path, str]]:
    samples = []
    for path in sorted(sample_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = path.parent.name
            samples.append((path, label))
    if not samples:
        raise FileNotFoundError(f"No images found under {sample_dir}")
    return samples


def normalize_label(label: str) -> str:
    aliases = {
        "deepal": "Deepal",
        "haval": "Haval",
        "mini": "Mini",
        "mercedes": "Mercedes-Benz",
        "mercedes-benz": "Mercedes-Benz",
    }
    return aliases.get(label, aliases.get(label.lower(), label))


def list_checkpoints(bundle_dir: Path, checkpoint: Path | None = None) -> list[Path]:
    if checkpoint is not None:
        checkpoint_path = (
            checkpoint if checkpoint.is_absolute() else bundle_dir / checkpoint
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return [checkpoint_path]

    checkpoints = sorted(bundle_dir.rglob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoints found under {bundle_dir}")
    return checkpoints


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"state_dict": checkpoint}


def masked_topk(
    probs: torch.Tensor,
    class_names: list[str],
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_probs = probs.clone()
    for label in DISALLOWED_PREDICTION_LABELS:
        if label in class_names:
            masked_probs[:, class_names.index(label)] = -1.0
    return masked_probs.topk(k=min(k, masked_probs.shape[1]), dim=1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    samples = list_samples(args.sample_dir)
    checkpoints = list_checkpoints(args.bundle_dir, args.checkpoint)

    prediction_rows = []
    summary_rows = []

    for index, checkpoint_path in enumerate(checkpoints, start=1):
        start = time.perf_counter()
        status = "ok"
        error = ""
        correct = 0
        total = len(samples)
        model_name = ""
        class_names = []
        try:
            checkpoint = load_checkpoint(checkpoint_path, device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model_name = infer_model_name(checkpoint_path, checkpoint)
            num_classes = int(checkpoint.get("num_classes", 26))
            dropout_p = float(checkpoint.get("dropout_p", 0.2))
            class_names = list(
                checkpoint.get("class_names", [str(i) for i in range(num_classes)])
            )
            model = build_model(model_name, num_classes, dropout_p).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            with torch.no_grad():
                for batch_start in range(0, len(samples), args.batch_size):
                    batch_samples = samples[batch_start : batch_start + args.batch_size]
                    sample_batch = torch.stack(
                        [load_image(path) for path, _label in batch_samples]
                    ).to(device)
                    logits = model(sample_batch)
                    probs = logits.softmax(dim=1)
                    top2_probs, top2_idxs = masked_topk(probs, class_names, k=2)
                    for row_idx, (image_path, true_label_raw) in enumerate(
                        batch_samples
                    ):
                        pred_idx = int(top2_idxs[row_idx, 0].cpu())
                        confidence = float(top2_probs[row_idx, 0].cpu())
                        second_idx = (
                            int(top2_idxs[row_idx, 1].cpu())
                            if top2_idxs.shape[1] > 1
                            else pred_idx
                        )
                        second_confidence = (
                            float(top2_probs[row_idx, 1].cpu())
                            if top2_probs.shape[1] > 1
                            else 0.0
                        )
                        pred_label = (
                            class_names[pred_idx]
                            if pred_idx < len(class_names)
                            else str(pred_idx)
                        )
                        second_label = (
                            class_names[second_idx]
                            if second_idx < len(class_names)
                            else str(second_idx)
                        )
                        true_label_normalized = normalize_label(true_label_raw)
                        true_in_model = true_label_normalized in class_names
                        is_correct = pred_label == true_label_normalized
                        correct += int(is_correct)
                        prediction_rows.append(
                            {
                                "checkpoint": checkpoint_path.relative_to(
                                    args.bundle_dir
                                ).as_posix(),
                                "model_name": model_name,
                                "image_path": image_path.relative_to(
                                    args.sample_dir
                                ).as_posix(),
                                "true_label_raw": true_label_raw,
                                "true_label_normalized": true_label_normalized,
                                "true_in_model": int(true_in_model),
                                "pred_label": pred_label,
                                "confidence": f"{confidence:.6f}",
                                "second_label": second_label,
                                "second_confidence": f"{second_confidence:.6f}",
                                "correct": int(is_correct),
                            }
                        )
        except Exception as exc:
            status = "error"
            error = repr(exc)
        known_rows = [
            row
            for row in prediction_rows
            if row["checkpoint"]
            == checkpoint_path.relative_to(args.bundle_dir).as_posix()
            and int(row.get("true_in_model", 0)) == 1
        ]
        known_correct = sum(int(row["correct"]) for row in known_rows)
        known_total = len(known_rows)
        seconds = time.perf_counter() - start
        summary_rows.append(
            {
                "checkpoint": checkpoint_path.relative_to(args.bundle_dir).as_posix(),
                "model_name": model_name,
                "status": status,
                "num_images": total,
                "known_label_images": known_total if status == "ok" else "",
                "correct": correct if status == "ok" else "",
                "acc": f"{(correct / total):.6f}" if status == "ok" and total else "",
                "known_label_correct": known_correct if status == "ok" else "",
                "known_label_acc": (
                    f"{(known_correct / known_total):.6f}"
                    if status == "ok" and known_total
                    else ""
                ),
                "seconds": f"{seconds:.3f}",
                "error": error,
            }
        )
        print(f"[{index}/{len(checkpoints)}] {checkpoint_path.name}: {status}")

    summary_path = args.output_dir / "pth_smoke_summary.csv"
    predictions_path = args.output_dir / "pth_smoke_predictions.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "checkpoint",
            "model_name",
            "image_path",
            "true_label_raw",
            "true_label_normalized",
            "true_in_model",
            "pred_label",
            "confidence",
            "second_label",
            "second_confidence",
            "correct",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()
