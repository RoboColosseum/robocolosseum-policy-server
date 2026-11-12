"""Traversability labels for the prepared Wuhan frames (BIT-3406).

255 = traversable: the driven strip, plus the ground SAM 2.1 large grows from it (minus detected obstacles)
  0 = not traversable: detected obstacles (Grounding DINO boxes -> SAM 2.1 masks), everything above the horizon,
      and what the original SAM-TP already calls non-traversable outside the grown ground
128 = unknown (ignored by the loss; the fine-tune keeps the original model's answer there)
Overexposed or flat frames are skipped.
"""
import json, os, sys
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

root, teacher_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
os.makedirs(out, exist_ok=True)
dev = 'cuda'
OBSTACLES = ('curb. step. stairs. wall. hedge. bush. bollard. bench. person. bicycle. motorcycle. scooter. car. '
             'pole. tree. planter. sign. barrier. traffic cone.')
dino_proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
dino = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base').to(dev).eval()
sam_proc = Sam2Processor.from_pretrained('facebook/sam2.1-hiera-large')
sam = Sam2Model.from_pretrained('facebook/sam2.1-hiera-large').to(dev).eval()
rows = [json.loads(l) for l in open(os.path.join(root, 'meta.jsonl'))]
rows = [r for r in rows if os.path.exists(os.path.join(teacher_dir, r['name'] + '.png'))]
if limit:
    rows = rows[:: max(1, len(rows) // limit)][:limit]
stats = {'labelled': 0, 'skipped_exposure': 0}
log = open(os.path.join(out, 'labels.jsonl'), 'a')


@torch.inference_mode()
def sam_masks(image, boxes=None, points=None, labels=None):
    kwargs = {}
    if boxes is not None:
        kwargs['input_boxes'] = [[b.tolist() for b in boxes]]
    if points is not None:
        kwargs['input_points'] = [[points.tolist()]]
        kwargs['input_labels'] = [[labels.tolist()]]
    inputs = sam_proc(images=image, return_tensors='pt', **kwargs).to(dev)
    outputs = sam(**inputs, multimask_output=points is not None)
    masks = sam_proc.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes'])[0]
    return masks.numpy().astype(bool), outputs.iou_scores.cpu().numpy()[0]


for k, r in enumerate(rows):
    name = r['name']
    bgr = cv2.imread(os.path.join(root, 'images', name + '.jpg'))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 215 or gray.std() < 18:
        stats['skipped_exposure'] += 1
        continue
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = gray.shape
    foot = cv2.imread(os.path.join(root, 'footprint', name + '.png'), cv2.IMREAD_GRAYSCALE) > 127
    teacher = cv2.imread(os.path.join(teacher_dir, name + '.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    with torch.inference_mode():
        inputs = dino_proc(images=image, text=OBSTACLES, return_tensors='pt').to(dev)
        det = dino_proc.post_process_grounded_object_detection(dino(**inputs), inputs.input_ids, threshold=0.30,
                                                               text_threshold=0.25, target_sizes=[(h, w)])[0]
    boxes = det['boxes'].cpu().numpy()
    names = list(det.get('text_labels', det.get('labels')))
    # Huge boxes (> 60% of the image) are usually "wall" or "stairs" swallowing the whole scene; keep them only above the horizon.
    obstacle = np.zeros((h, w), bool)
    kept = []
    for box, label in zip(boxes, names):
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw * bh > 0.6 * w * h:
            continue
        kept.append((box, label))
    if kept:
        masks, _ = sam_masks(image, boxes=np.array([b for b, _ in kept]))
        for m in masks:
            obstacle |= m[0] if m.ndim == 3 else m
    # Ground grown from the driven strip: positive points along it, negative points on obstacles and the sky.
    ys, xs = np.nonzero(foot)
    order = np.argsort(ys)
    pick = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    pos = np.stack([xs[pick], ys[pick]], 1)
    neg = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b, _ in kept[:6]] + [[w * 0.25, h * 0.15], [w * 0.75, h * 0.15]]
    points = np.concatenate([pos, np.array(neg)]).astype(np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    masks, scores = sam_masks(image, points=points, labels=labels)
    masks = masks[0] if masks.ndim == 4 else masks
    best, best_score = None, -1.0
    horizon = int(h / 2) - 4
    for m, s in zip(masks, np.ravel(scores)):
        cover = (m & foot).sum() / max(1, foot.sum())
        above = m[:horizon].mean()
        score = cover - 3.0 * above + 0.1 * float(s)
        if cover > 0.85 and score > best_score:
            best, best_score = m, score
    ground = best if best is not None else np.zeros_like(foot)
    ground[:horizon] = False
    label = np.full((h, w), 128, np.uint8)
    label[teacher < 0.25] = 0
    label[:horizon] = 0
    label[obstacle] = 0
    label[ground & ~obstacle] = 255
    label[foot] = 255
    cv2.imwrite(os.path.join(out, name + '.png'), label)
    log.write(json.dumps({'name': name, 'obstacles': [str(l) for _, l in kept], 'ground': best is not None,
                          'traversable': round(float((label == 255).mean()), 3), 'unknown': round(float((label == 128).mean()), 3)}) + '\n')
    stats['labelled'] += 1
    if k % 250 == 0:
        print(k, len(rows), stats, flush=True)
print('done', stats, flush=True)
