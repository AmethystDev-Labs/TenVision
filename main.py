import cv2
import numpy as np
import sys

IMG_PATH = "images/0.png"
OUT_PATH = "result_demo.png"


def normalize_mask(binary_mask: np.ndarray, canvas_size: int = 48, symbol_size: int = 34) -> np.ndarray | None:
    ys, xs = np.where(binary_mask > 0)
    if xs.size == 0:
        return None

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    crop = binary_mask[y1:y2, x1:x2]

    h, w = crop.shape
    scale = symbol_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    oy = (canvas_size - new_h) // 2
    ox = (canvas_size - new_w) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    h, w = mask.shape
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated


def crop_foreground(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return mask[y1:y2, x1:x2]


def match_cost(query: np.ndarray, candidate: np.ndarray, allow_rotate: bool = True) -> float:
    diff = cv2.absdiff(query, candidate)
    best = float(np.sum(diff) / 255.0)
    if not allow_rotate:
        return best

    for angle in (-60, -45, -30, -20, -10, 10, 20, 30, 45, 60, 90, -90):
        rotated = rotate_mask(query, angle)
        score = float(np.sum(cv2.absdiff(rotated, candidate)) / 255.0)
        if score < best:
            best = score
    return best


def locate_with_template(query_mask: np.ndarray, main_mask: np.ndarray) -> tuple[float, tuple[int, int] | None]:
    query_crop = crop_foreground(query_mask)
    if query_crop is None:
        return -1.0, None

    qh, qw = query_crop.shape
    if min(qh, qw) < 10:
        return -1.0, None

    best_score = -1.0
    best_center = None
    scales = np.linspace(1.1, 3.4, 16)
    angles = range(-90, 91, 10)

    for scale in scales:
        new_w = max(8, int(round(qw * scale)))
        new_h = max(8, int(round(qh * scale)))
        base = cv2.resize(query_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        for angle in angles:
            rotated = rotate_mask(base, angle)
            rotated = crop_foreground(rotated)
            if rotated is None:
                continue
            if rotated.shape[0] >= main_mask.shape[0] or rotated.shape[1] >= main_mask.shape[1]:
                continue
            if np.count_nonzero(rotated) < 40:
                continue

            result = cv2.matchTemplate(main_mask, rotated, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score > best_score:
                cx = loc[0] + rotated.shape[1] // 2
                cy = loc[1] + rotated.shape[0] // 2
                best_score = float(score)
                best_center = (cx, cy)

    return best_score, best_center


if len(sys.argv) >= 2:
    IMG_PATH = sys.argv[1]
if len(sys.argv) >= 3:
    OUT_PATH = sys.argv[2]

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(IMG_PATH)

height, width = img.shape[:2]

# 1) 自动定位顶部灰色题目条（包含 3 个小格）
top_region = img[int(0.06 * height):int(0.25 * height), int(0.18 * width):int(0.82 * width)]
top_gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
gray_mask = ((top_gray > 110) & (top_gray < 220)).astype(np.uint8) * 255
gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_mask, connectivity=8)
strip_box = None
best_area = -1
top_h = top_region.shape[0]
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if not (80 <= w <= 220 and 18 <= h <= 40):
        continue
    if y > int(0.7 * top_h):
        continue
    if area > best_area:
        best_area = area
        strip_box = (x, y, w, h)

if strip_box is None:
    raise RuntimeError("没有找到顶部题目灰框，请调 gray_mask 阈值。")

sx, sy, sw, sh = strip_box
strip_roi = top_region[sy:sy + sh, sx:sx + sw]
strip_roi_gray = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2GRAY)
query_cells = np.array_split(strip_roi_gray, 3, axis=1)

query_templates = []
query_raw_masks = []
for cell in query_cells:
    _, cell_bw = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cell_bw[:2, :] = 0
    cell_bw[-2:, :] = 0
    cell_bw[:, :2] = 0
    cell_bw[:, -2:] = 0
    cell_bw = cv2.morphologyEx(cell_bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    query_raw_masks.append(cell_bw)
    template = normalize_mask(cell_bw)
    query_templates.append(template)

# 2) 主图里提取黑色符号候选
main_y1, main_y2 = int(0.20 * height), int(0.88 * height)
main_x1, main_x2 = int(0.00 * width), int(0.96 * width)
main_roi = img[main_y1:main_y2, main_x1:main_x2]
main_gray = cv2.cvtColor(main_roi, cv2.COLOR_BGR2GRAY)

symbol_bw = (main_gray < 60).astype(np.uint8) * 255
symbol_bw = cv2.morphologyEx(symbol_bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
template_main_bw = (main_gray < 80).astype(np.uint8) * 255

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(symbol_bw, connectivity=8)
candidates = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area < 50 or area > 6000:
        continue
    if w < 10 or h < 10:
        continue
    if w / max(h, 1) > 3.0 or h / max(w, 1) > 3.0:
        continue

    component_mask = np.where(labels[y:y + h, x:x + w] == i, 255, 0).astype(np.uint8)
    normalized = normalize_mask(component_mask)
    if normalized is None:
        continue

    candidates.append(
        {
            "center": (int(centroids[i][0]), int(centroids[i][1])),
            "bbox": (x, y, w, h),
            "norm": normalized,
        }
    )

if not candidates:
    raise RuntimeError("主图没有提取到候选符号，请调黑色阈值。")

# 3) 按题目顺序做一对一匹配
used = set()
ordered_points = []
base_scores = []

for template in query_templates:
    if template is None:
        ordered_points.append(None)
        base_scores.append(float("inf"))
        continue

    best_idx = -1
    best_score = float("inf")
    for idx, candidate in enumerate(candidates):
        if idx in used:
            continue
        score = match_cost(template, candidate["norm"], allow_rotate=True)
        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx >= 0:
        used.add(best_idx)
        cx, cy = candidates[best_idx]["center"]
        ordered_points.append((main_x1 + cx, main_y1 + cy))
        base_scores.append(best_score)
    else:
        ordered_points.append(None)
        base_scores.append(float("inf"))

# 3.5) 模板匹配校正：复杂符号走旋转+缩放模板搜索
for i, raw_mask in enumerate(query_raw_masks):
    # 基础匹配已经较好时，不做模板校正，避免把 4/9 这类简单符号改坏
    if ordered_points[i] is not None and base_scores[i] < 280:
        continue

    score, center = locate_with_template(raw_mask, template_main_bw)
    if center is None:
        continue
    if score < 0.60:
        continue

    new_point = (main_x1 + center[0], main_y1 + center[1])
    old_point = ordered_points[i]

    # 避免不同序号落在几乎同一点
    too_close = False
    for j, point in enumerate(ordered_points):
        if j == i or point is None:
            continue
        distance = ((point[0] - new_point[0]) ** 2 + (point[1] - new_point[1]) ** 2) ** 0.5
        if distance < 26:
            too_close = True
            break

    if too_close:
        continue

    # 有旧点时，仅在基础匹配不稳定时才替换
    if old_point is not None and base_scores[i] < 340:
        continue

    ordered_points[i] = new_point

# 4) 可视化
vis = img.copy()
cv2.rectangle(
    vis,
    (int(0.18 * width) + sx, int(0.06 * height) + sy),
    (int(0.18 * width) + sx + sw, int(0.06 * height) + sy + sh),
    (255, 0, 0),
    2,
)

for candidate in candidates:
    x, y, w, h = candidate["bbox"]
    cv2.rectangle(vis, (main_x1 + x, main_y1 + y), (main_x1 + x + w, main_y1 + y + h), (0, 255, 255), 1)

for idx, point in enumerate(ordered_points, start=1):
    if point is None:
        continue
    cv2.circle(vis, point, 16, (0, 255, 0), 2)
    cv2.putText(vis, str(idx), (point[0] - 6, point[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite(OUT_PATH, vis)
print("点击顺序坐标:", ordered_points)
print("候选数:", len(candidates))
print("已保存:", OUT_PATH)
