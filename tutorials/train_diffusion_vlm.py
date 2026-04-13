"""
Training script for ViLDiffusionVLM.

Two-stage recipe:
  Stage 1 — CC3M pretraining  (~50k steps, encoder frozen)
  Stage 2 — COCO fine-tuning  (~20-30k steps, encoder unfrozen at lower lr)

Run on Kaggle (single T4/P100):
  !python tutorials/train_diffusion_vlm.py \
      --stage 1 \
      --data_root /kaggle/input/cc3m \
      --encoder_path /kaggle/input/vil-encoder/vil2_tiny16_e400_in1k.th \
      --output_dir /kaggle/working/checkpoints \
      --steps 50000

For stage 2 (COCO fine-tune), pass --stage 2 --resume <checkpoint>.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from caption_lstm import ViLDiffusionVLM, ViLDiffusionConfig


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CocoCaptionsDataset(Dataset):
    """COCO Captions — expects the standard annotations JSON layout."""

    def __init__(self, root: str, ann_file: str, transform=None):
        self.root = Path(root)
        self.transform = transform

        with open(ann_file) as f:
            data = json.load(f)

        id2file = {img['id']: img['file_name'] for img in data['images']}
        self.samples = [
            (id2file[ann['image_id']], ann['caption'])
            for ann in data['annotations']
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        img = Image.open(self.root / fname).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, caption


class CC3MDataset(Dataset):
    """
    CC3M / CC12M — expects a TSV file with columns: image_path<TAB>caption
    or a folder of (image, .txt) pairs.
    Adjust __getitem__ to match your local layout.
    """

    def __init__(self, tsv_path: str, image_root: str, transform=None):
        self.image_root = Path(image_root)
        self.transform = transform
        self.samples = []
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    self.samples.append(tuple(parts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        try:
            img = Image.open(self.image_root / fname).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, caption


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model weights for stable inference."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module):
        """Copy EMA weights into model (for inference)."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module, original: dict):
        """Restore original weights after an EMA inference pass."""
        for name, p in model.named_parameters():
            p.data.copy_(original[name])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_transform(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def build_optimizer(model: ViLDiffusionVLM, lr: float, weight_decay: float, encoder_unfrozen: bool):
    """
    Two-group optimizer: lower lr for encoder params when unfrozen.
    """
    if encoder_unfrozen:
        encoder_params = list(model.encoder_parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in model.parameters()
                        if id(p) not in encoder_ids and p.requires_grad]
        param_groups = [
            {'params': other_params,  'lr': lr},
            {'params': encoder_params, 'lr': lr * 0.1},
        ]
    else:
        param_groups = [{'params': model.denoiser_parameters(), 'lr': lr}]

    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=weight_decay)


def cosine_lr(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        # encoder group has its own lr_scale baked in; preserve ratio
        scale = g.get('_scale', 1.0)
        g['lr'] = lr * scale


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = ViLDiffusionConfig(
        encoder_pretrained_path=args.encoder_path,
        freeze_encoder=(args.stage == 1),
        model_dim=512,
        num_denoiser_blocks=6,
        max_text_length=32,
        p_uncond=0.1,
        num_inference_steps=20,
        guidance_scale=2.0,
    )
    model = ViLDiffusionVLM(config).to(device)
    print(f"Trainable parameters: {model.num_parameters() / 1e6:.1f}M")

    # Resume from checkpoint if given
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    # If stage 2, unfreeze encoder
    if args.stage == 2:
        model.unfreeze_encoder()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    transform_train = get_transform(is_train=True)
    transform_val = get_transform(is_train=False)

    if args.stage == 1:
        # CC3M pretraining
        train_dataset = CC3MDataset(args.cc3m_tsv, args.data_root, transform_train)
    else:
        # COCO fine-tuning
        train_dataset = CocoCaptionsDataset(
            args.data_root,
            os.path.join(args.data_root, 'annotations/captions_train2017.json'),
            transform_train,
        )
        val_dataset = CocoCaptionsDataset(
            args.data_root,
            os.path.join(args.data_root, 'annotations/captions_val2017.json'),
            transform_val,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Optimiser, scheduler, scaler
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, args.lr, args.weight_decay,
                                encoder_unfrozen=(args.stage == 2))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    ema = EMA(model, decay=0.9999)

    # Mark encoder group lr scale so set_lr can preserve the ratio
    if args.stage == 2:
        optimizer.param_groups[1]['_scale'] = 0.1  # encoder uses 10% of main lr

    warmup_steps = args.warmup_steps
    total_steps = args.steps

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad()

    loader_iter = iter(train_loader)
    global_step = start_step
    accum_loss = 0.0
    log_every = 100

    pbar = tqdm(range(start_step, total_steps), initial=start_step, total=total_steps)
    for step in pbar:
        # Cosine LR
        lr = cosine_lr(step, warmup_steps, total_steps, args.lr, args.lr * 0.01)
        for g in optimizer.param_groups:
            scale = g.get('_scale', 1.0)
            g['lr'] = lr * scale

        # Fetch batch (cycling through loader)
        try:
            images, captions = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            images, captions = next(loader_iter)

        images = images.to(device, non_blocking=True)

        # Mixed-precision forward
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            out = model(images, captions)
            loss = out['loss'] / args.grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        # Logging
        if (step + 1) % log_every == 0:
            pbar.set_postfix({
                'loss': f"{accum_loss * args.grad_accum / log_every:.4f}",
                'mask%': f"{out['mask_fraction']:.2f}",
                'acc': f"{out['masked_accuracy']:.3f}",
                'lr': f"{lr:.2e}",
            })
            accum_loss = 0.0

        # Checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"step_{step + 1}.pt"
            torch.save({
                'model': model.state_dict(),
                'ema': ema.shadow,
                'optimizer': optimizer.state_dict(),
                'step': step + 1,
                'config': config,
            }, ckpt_path)
            print(f"\nSaved checkpoint → {ckpt_path}")

        # Quick validation sample (COCO stage 2 only)
        if args.stage == 2 and (step + 1) % args.val_every == 0:
            model.eval()
            val_batch = next(iter(DataLoader(val_dataset, batch_size=4,
                                             num_workers=0, shuffle=True)))
            val_images, val_gt = val_batch
            val_images = val_images.to(device)
            generated = model.generate(val_images, num_steps=20)
            print("\n--- Validation samples ---")
            for gt, pred in zip(val_gt[:4], generated[:4]):
                print(f"  GT  : {gt}")
                print(f"  PRED: {pred}")
                print()
            model.train()

        global_step += 1

    # Final checkpoint
    torch.save({
        'model': model.state_dict(),
        'ema': ema.shadow,
        'step': total_steps,
        'config': config,
    }, output_dir / 'final.pt')
    print(f"Training complete. Saved to {output_dir / 'final.pt'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, default=1, choices=[1, 2],
                   help='1=CC3M pretrain (encoder frozen), 2=COCO finetune')
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to dataset root (COCO or CC3M images)')
    p.add_argument('--cc3m_tsv', type=str, default='',
                   help='Path to CC3M TSV file (stage 1 only)')
    p.add_argument('--encoder_path', type=str, default=None,
                   help='Path to pretrained VisionLSTM weights (.th)')
    p.add_argument('--output_dir', type=str, default='checkpoints')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--steps', type=int, default=50000,
                   help='Total training steps')
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=16,
                   help='Per-GPU batch size (use grad_accum to reach effective 512)')
    p.add_argument('--grad_accum', type=int, default=32,
                   help='Gradient accumulation steps (effective_bs = batch_size * grad_accum)')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--val_every', type=int, default=2000)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
