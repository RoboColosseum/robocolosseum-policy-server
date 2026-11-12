"""Contact sheet: images with a mask overlay. python sheet.py PREPARED MASKDIR OUT.jpg [n] [seed]"""
import json, random, sys, os
import cv2, numpy as np
root, maskdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 24
random.seed(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
rows = [json.loads(l) for l in open(os.path.join(root, 'meta.jsonl'))]
rows = [r for r in rows if os.path.exists(os.path.join(maskdir, r['name'] + '.png'))]
pick = random.sample(rows, min(n, len(rows)))
tiles = []
for r in pick:
    im = cv2.imread(os.path.join(root, 'images', r['name'] + '.jpg'))
    m = cv2.imread(os.path.join(maskdir, r['name'] + '.png'), cv2.IMREAD_GRAYSCALE)
    over = im.copy()
    if m is not None:
        over[m > 200] = (0.45 * over[m > 200] + 0.55 * np.array([0, 220, 0])).astype(np.uint8)
        over[(m > 100) & (m <= 200)] = (0.6 * over[(m > 100) & (m <= 200)] + 0.4 * np.array([0, 200, 255])).astype(np.uint8)
        over[m < 50] = (0.65 * over[m < 50] + 0.35 * np.array([0, 0, 220])).astype(np.uint8) if 'label' in maskdir else over[m < 50]
    cv2.putText(over, f"{r['name']} {r['split']} {r['path_m']}m", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(cv2.resize(over, (384, 216)))
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
