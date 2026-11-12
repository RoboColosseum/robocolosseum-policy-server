"""Fine-tune SAM-TP (GeNIE's traversability model) on the Wuhan labels (BIT-3406).

Starts from the authors' checkpoint_2.pt and trains every part, as their freezeNoneNone config does: the Hiera-tiny
image encoder (lower learning rate), the learned prompt and the mask decoder. The forward pass is the one GeNIE
runs (SAM2Transforms, no_mem_embed, the learned prompt, single mask, logits resized to the frame).

Targets come from refine.py: 0-254 is the traversability target (x/254; soft inside the clearance margin), 255 is
unknown. Loss on known pixels: 20 x focal + 1 x dice (the authors' weights, with soft targets). On unknown pixels:
binary cross-entropy towards the original model's probability, so the fine-tune keeps its answer there.

    python train.py --data prepared --labels prepared/labels --teacher prepared/teacher --ckpt weights/checkpoint_2.pt --out runs/ft1
"""
import argparse, json, math, os, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sam2
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

H, W = 288, 512
BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]


class Frames(torch.utils.data.Dataset):
    def __init__(self, root, labels, teacher, names, train):
        self.root, self.labels, self.teacher, self.names, self.train = root, labels, teacher, names, train
        self.transform = SAM2Transforms(resolution=1024, mask_threshold=0.0)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root, 'images', name + '.jpg')), cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.labels, name + '.png'), cv2.IMREAD_GRAYSCALE)
        teacher = cv2.imread(os.path.join(self.teacher, name + '.png'), cv2.IMREAD_GRAYSCALE)
        foot = cv2.imread(os.path.join(self.root, 'footprint', name + '.png'), cv2.IMREAD_GRAYSCALE)
        if self.train:
            if random.random() < 0.5:
                rgb, label, teacher, foot = rgb[:, ::-1], label[:, ::-1], teacher[:, ::-1], foot[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.85, 1.15)
            hsv[..., 2] = hsv[..., 2] * random.uniform(0.8, 1.2) + random.uniform(-12, 12)
            rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        image = self.transform(np.ascontiguousarray(rgb))
        return (image, torch.from_numpy(np.ascontiguousarray(label)), torch.from_numpy(np.ascontiguousarray(teacher)),
                torch.from_numpy(np.ascontiguousarray(foot)))


def forward(model, images):
    """Traversability logits (B, H, W) for a batch of transformed 1024x1024 images, as GeNIE computes them."""
    b = images.shape[0]
    backbone_out = model.forward_image(images)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [feat.permute(1, 2, 0).view(b, -1, *size)
             for feat, size in zip(vision_feats[::-1], BB_FEAT_SIZES[::-1])][::-1]
    points = torch.zeros(b, 3, 2, device=images.device)
    labels = torch.ones(b, 3, dtype=torch.int32, device=images.device)
    sparse, dense = model.sam_prompt_encoder(points=(points, labels), boxes=None, masks=None)
    low_res, _, _, _ = model.sam_mask_decoder(image_embeddings=feats[-1], image_pe=model.sam_prompt_encoder.get_dense_pe(),
                                              sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                              multimask_output=False, repeat_image=False, high_res_features=feats[:-1])
    return F.interpolate(low_res.float(), (H, W), mode='bilinear', align_corners=False)[:, 0]


def loss_fn(logits, label, teacher, focal_w=20.0, dice_w=1.0, distill_w=0.5):
    known = label != 255
    target = label.float().clamp(max=254) / 254.0
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = p * target + (1 - p) * (1 - target)
    focal = (0.25 * target + 0.75 * (1 - target)) * (1 - pt) ** 2 * ce
    focal = (focal * known).sum() / known.sum().clamp(min=1)
    inter = (p * target * known).sum((1, 2))
    dice = 1 - (2 * inter + 1) / ((p * known).sum((1, 2)) + (target * known).sum((1, 2)) + 1)
    unknown = ~known
    distill = F.binary_cross_entropy_with_logits(logits, teacher.float() / 255.0, reduction='none')
    distill = (distill * unknown).sum() / unknown.sum().clamp(min=1)
    total = focal_w * focal + dice_w * dice.mean() + distill_w * distill
    return total, {'focal': focal.item(), 'dice': dice.mean().item(), 'distill': distill.item()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Per-pixel metrics on the held-out episodes (threshold 0.5)."""
    model.eval()
    sums = dict(tp=0, fp=0, fn=0, foot=0, foot_hit=0, neg=0, neg_hit=0, unk=0, unk_agree=0, margin=0, margin_p=0.0,
                margin_t=0.0, loss=0.0, batches=0)
    for images, label, teacher, foot in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = forward(model, images.to(device))
        loss, _ = loss_fn(logits, label.to(device), teacher.to(device))
        sums['loss'] += loss.item(); sums['batches'] += 1
        prob = torch.sigmoid(logits).float().cpu()
        pred = prob > 0.5
        pos, neg, unk = label == 254, label == 0, label == 255
        margin = (label > 0) & (label < 254)
        sums['margin'] += int(margin.sum()); sums['margin_p'] += float(prob[margin].sum()); sums['margin_t'] += float((label[margin].float() / 254).sum())
        sums['tp'] += int((pred & pos).sum()); sums['fp'] += int((pred & neg).sum()); sums['fn'] += int((~pred & pos).sum())
        f = foot > 127
        sums['foot'] += int(f.sum()); sums['foot_hit'] += int((pred & f).sum())
        sums['neg'] += int(neg.sum()); sums['neg_hit'] += int((pred & neg).sum())
        sums['unk'] += int(unk.sum()); sums['unk_agree'] += int((pred == (teacher > 127))[unk].sum())
    model.train()
    return {'iou': sums['tp'] / max(1, sums['tp'] + sums['fp'] + sums['fn']),
            'driven_path_recall': sums['foot_hit'] / max(1, sums['foot']),
            'obstacle_false_traversable': sums['neg_hit'] / max(1, sums['neg']),
            'teacher_agreement_unknown': sums['unk_agree'] / max(1, sums['unk']),
            'margin_mean_prob': sums['margin_p'] / max(1, sums['margin']), 'margin_mean_target': sums['margin_t'] / max(1, sums['margin']),
            'val_loss': sums['loss'] / max(1, sums['batches'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--teacher', required=True)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5); ap.add_argument('--vision-lr', type=float, default=2e-6)
    ap.add_argument('--distill', type=float, default=0.5); ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    rows = [json.loads(l) for l in open(os.path.join(args.data, 'meta.jsonl'))]
    have = {p.stem for p in Path(args.labels).glob('*.png')}
    train = [r['name'] for r in rows if r['split'] == 'train' and r['name'] in have]
    val = [r['name'] for r in rows if r['split'] == 'val' and r['name'] in have]
    print(f'train {len(train)} val {len(val)}', flush=True)
    config = Path(sam2.__file__).parent / 'configs' / 'sam2.1_inference_tiny' / 'sam2.1_custom2.yaml'
    model = build_sam2(str(config), args.ckpt, device=device, mode='train')
    model.train()
    mk = lambda names, tr: torch.utils.data.DataLoader(Frames(args.data, args.labels, args.teacher, names, tr), batch_size=args.batch,
                                                       shuffle=tr, num_workers=6, drop_last=tr, pin_memory=True, persistent_workers=True)
    train_loader, val_loader = mk(train, True), mk(val, False)
    vision = [p for n, p in model.named_parameters() if n.startswith('image_encoder')]
    heads = [p for n, p in model.named_parameters() if not n.startswith('image_encoder')]
    opt = torch.optim.AdamW([{'params': vision, 'lr': args.vision_lr}, {'params': heads, 'lr': args.lr}], weight_decay=0.1)
    steps = args.epochs * len(train_loader)
    base = [g['lr'] for g in opt.param_groups]
    history = []
    metrics = evaluate(model, val_loader, device)
    print('epoch 0 (original SAM-TP)', json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
    history.append({'epoch': 0, **metrics})
    best = -1.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        started, running = time.time(), []
        for images, label, teacher, foot in train_loader:
            for g, b0 in zip(opt.param_groups, base):  # cosine to a tenth, like the authors' schedule
                g['lr'] = b0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = forward(model, images.to(device, non_blocking=True))
            loss, parts = loss_fn(logits, label.to(device), teacher.to(device), distill_w=args.distill)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            running.append(loss.item()); step += 1
            if step % 100 == 0:
                print(f'epoch {epoch} step {step}/{steps} loss {np.mean(running[-100:]):.4f} {json.dumps({k: round(v, 4) for k, v in parts.items()})}', flush=True)
        metrics = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'loss': float(np.mean(running)), **metrics})
        print(f'epoch {epoch} ({time.time() - started:.0f} s)', json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
        score = -metrics['val_loss']
        torch.save({'model': model.state_dict()}, os.path.join(args.out, 'last.pt'))
        if score > best:
            best = score
            torch.save({'model': model.state_dict()}, os.path.join(args.out, 'best.pt'))
            print(f'  saved best (val loss {-score:.4f})', flush=True)
        json.dump(history, open(os.path.join(args.out, 'history.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
