import argparse
from config import Config, set_seeds
from finetune import train_main
from resume_train import resume_main

def parse_args():
    p = argparse.ArgumentParser("RDD Project Trainer")
    p.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    p.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (.pth) when --resume is used")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--data-dir", type=str, default=None, help="Override base data dir (default: data/processed)")
    p.add_argument("--ckpt-dir", type=str, default=None, help="Override checkpoints dir")
    p.add_argument("--log-dir", type=str, default=None, help="Override TensorBoard log dir")
    return p.parse_args()

def override_cfg(cfg: Config, args):
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.data_dir is not None:
        cfg.base_path = args.data_dir  
        cfg.__post_init__()
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir

def main():
    args = parse_args()
    cfg = Config()
    override_cfg(cfg, args)
    set_seeds(cfg.seed)

    if args.resume:
        if not args.checkpoint:
            raise ValueError("Please provide --checkpoint when using --resume")
        resume_main(cfg, args.checkpoint)
    else:
        train_main(cfg)

if __name__ == "__main__":
    main()
