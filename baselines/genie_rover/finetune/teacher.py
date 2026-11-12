"""The original SAM-TP's traversability (sigmoid of its logits, 0-255 PNG) for every prepared frame, as GeNIE calls it."""
import json, os, sys
from pathlib import Path
import cv2, numpy as np, torch
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

root, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
config = Path(sam2.__file__).parent / 'configs' / 'sam2.1_inference_tiny' / 'sam2.1_custom2.yaml'
predictor = SAM2ImagePredictor(build_sam2(str(config), ckpt, device='cuda'), mask_threshold=0.0)
rows = [json.loads(l) for l in open(os.path.join(root, 'meta.jsonl'))]
for k, r in enumerate(rows):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(root, 'images', r['name'] + '.jpg')), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    points = np.array([[0, h - 1], [w - 1, h - 1], [(w - 1) // 2, h - 1]], np.float32)
    predictor.reset_predictor(); predictor.set_image(rgb)
    masks, _, _ = predictor.predict(point_coords=points, point_labels=np.ones(3, np.int32), multimask_output=False,
                                    return_logits=True, normalize_coords=False)
    prob = 1.0 / (1.0 + np.exp(-np.asarray(masks[0], np.float32)))
    cv2.imwrite(os.path.join(out, r['name'] + '.png'), (prob * 255).round().astype(np.uint8))
    if k % 500 == 0:
        print(k, len(rows), flush=True)
print('done', flush=True)
