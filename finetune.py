from pathlib import Path
from typing import Dict, Any, Tuple, List
import time
import gc
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from config import Config, set_seeds
from load_dataset import load_metadata
from dataloader import get_dataloaders

class SmoothedValue:
    def __init__(self):
        self.total = 0.0
        self.count = 0
    def update(self, value: float):
        self.total += float(value)
        self.count += 1
    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

class MetricLogger:
    def __init__(self):
        self.meters = {"loss": SmoothedValue(), "lr": SmoothedValue()}
    def log_every(self, iterable, print_freq: int, header: str):
        for i, obj in enumerate(iterable):
            if i % max(print_freq, 1) == 0:
                print(header, f"[{i}]")
            yield i, obj
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            if torch.is_tensor(v):
                v = v.item()
            self.meters[k].update(v)
def build_model(num_classes: int):
    model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
def _denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_predictions(images, outputs, targets, int_to_class, writer: SummaryWriter, epoch: int):
    fig, axs = plt.subplots(nrows=len(images), ncols=1, figsize=(12, 8 * len(images)))
    if len(images) == 1:
        axs = [axs]
    for i, (img_t, out, tgt) in enumerate(zip(images, outputs, targets)):
        ax = axs[i]
        img_np = _denormalize_image(img_t)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        for box, label in zip(tgt['boxes'].cpu(), tgt['labels'].cpu()):
            x1, y1, x2, y2 = box
            name = int_to_class.get(label.item(), "UNK")
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, max(0, y1 - 16)), f"GT: {name}", fill="green", font=font)
        scores = out['scores'].cpu() if 'scores' in out else None
        keep = scores > 0.5 if scores is not None else torch.ones(len(out['boxes']), dtype=torch.bool)
        for box, label, score in zip(out['boxes'].cpu()[keep], out['labels'].cpu()[keep], out['scores'].cpu()[keep]):
            x1, y1, x2, y2 = box
            name = int_to_class.get(label.item(), "UNK")
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y2 + 2), f"Pred: {name} ({float(score):.2f})", fill="red", font=font)

        ax.imshow(img_pil)
        ax.axis("off")
        ax.set_title(f"Validation Sample {i}")

    plt.tight_layout()
    writer.add_figure('Validation/Predictions', fig, global_step=epoch)
    plt.close(fig)
def train_one_epoch(model, data_loader, device, optimizer, print_freq, epoch, writer, scaler=None):
    model.train()
    logger = MetricLogger()

    for i, (batch) in logger.log_every(enumerate(data_loader), print_freq, f"Training Epoch {epoch}:"):
        _, (images, targets) = i, batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(scaler is not None and device == torch.device("cuda"))):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None and device.type == "cuda":
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        logger.update(loss=losses, lr=optimizer.param_groups[0]["lr"])
        writer.add_scalar("Loss/train_total_batch", losses.item(), epoch * len(data_loader) + i)
        for k, v in loss_dict.items():
            writer.add_scalar(f"Loss/train_{k}_batch", v.item(), epoch * len(data_loader) + i)

    avg_loss = logger.meters["loss"].global_avg
    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/avg_train_epoch", avg_loss, epoch)

def evaluate(model, data_loader, device, epoch, writer, int_to_class):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox").to(device)
    samples = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            preds = [{"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"]} for o in outputs]
            targs = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            metric.update(preds, targs)

            if len(samples) < 5:
                for img, out, tgt in zip(images, outputs, targets):
                    samples.append((img, out, tgt))
                    if len(samples) >= 5:
                        break

    results = metric.compute()
    print("--- Validation mAP ---")
    print(results)
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                writer.add_scalar(f"mAP/{k}", v.item(), epoch)
            else:
                for idx, value in enumerate(v):
                    cname = int_to_class.get(idx, f"class_{idx}")
                    if k == "map_per_class":
                        writer.add_scalar(f"mAP_by_Class/{cname}", value.item(), epoch)
                    else:
                        writer.add_scalar(f"mAP_Array/{k}_{idx}", value.item(), epoch)

    if samples:
        visualize_predictions(
            [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples],
            int_to_class, writer, epoch
        )
    return results

def train_main(cfg: Config):
    set_seeds(cfg.seed)
    train_df_cleaned, class_to_int, int_to_class, _ = load_metadata(cfg.train_csv)
    train_loader, val_loader = get_dataloaders(
        cfg.train_csv, cfg.val_csv, cfg.train_img_dir, cfg.val_img_dir,
        class_to_int, cfg.batch_size
    )

    num_classes = 9
    device = torch.device(cfg.device)
    model = build_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.num_epochs // 2, gamma=cfg.step_lr_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    writer = SummaryWriter(log_dir=cfg.log_dir)
    best_map = float("-inf")

    start = time.time()
    try:
        for epoch in range(cfg.num_epochs):
            torch.cuda.empty_cache(); gc.collect()

            train_one_epoch(model, train_loader, device, optimizer, print_freq=50, epoch=epoch, writer=writer, scaler=scaler)
            lr_scheduler.step()
            results = evaluate(model, val_loader, device, epoch, writer, int_to_class)
            current_map = results['map'].item() if isinstance(results['map'], torch.Tensor) else float(results['map'])

            if current_map > best_map:
                best_map = current_map
                ckpt_path = Path(cfg.ckpt_dir, f"best_model_checkpoint_epoch_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_map": best_map,
                    "scaler": scaler.state_dict() if scaler is not None else None
                }, ckpt_path)
                print(f"New best mAP: {best_map:.4f} -> saved: {ckpt_path}")

            print(f"Epoch {epoch} done — current mAP: {current_map:.4f} | best: {best_map:.4f}\n")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        mins = (time.time() - start) / 60.0
        print(f"Training complete. Duration: {mins:.2f} min | Best mAP: {best_map:.4f}")
        print(f"Checkpoints saved in: {cfg.ckpt_dir}")
        writer.close()

    return {"best_map": best_map}
