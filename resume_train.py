from pathlib import Path
import time
import gc
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import Config, set_seeds
from load_dataset import load_metadata
from dataloader import get_dataloaders
from finetune import build_model, train_one_epoch, evaluate

def resume_main(cfg: Config, checkpoint_path: str):
    set_seeds(cfg.seed)
    _, class_to_int, int_to_class, _ = load_metadata(cfg.train_csv)
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

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    else:
        print("Warning: no scaler in checkpoint; continuing with fresh scaler.")

    epoch_start = ckpt["epoch"] + 1
    best_map = ckpt.get("best_map", float("-inf"))

    writer = SummaryWriter(log_dir=cfg.log_dir)
    start = time.time()

    try:
        for epoch in range(epoch_start, cfg.num_epochs):
            torch.cuda.empty_cache(); gc.collect()

            train_one_epoch(model, train_loader, device, optimizer, print_freq=50, epoch=epoch, writer=writer, scaler=scaler)
            lr_scheduler.step()
            results = evaluate(model, val_loader, device, epoch, writer, int_to_class)
            current_map = results['map'].item() if isinstance(results['map'], torch.Tensor) else float(results['map'])

            if current_map > best_map:
                best_map = current_map
                new_path = Path(cfg.ckpt_dir, f"best_model_checkpoint_epoch_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_map": best_map,
                    "scaler": scaler.state_dict() if scaler is not None else None
                }, new_path)
                print(f"New best mAP: {best_map:.4f} -> saved: {new_path}")

            print(f"Epoch {epoch} done — current mAP: {current_map:.4f} | best: {best_map:.4f}\n")

    except KeyboardInterrupt:
        print("Resumed training interrupted by user.")
    finally:
        mins = (time.time() - start) / 60.0
        print(f"Resume complete. Duration: {mins:.2f} min | Best mAP: {best_map:.4f}")
        writer.close()

    return {"best_map": best_map}
