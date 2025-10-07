import time, collections, json, os
from pathlib import Path

import numpy as np
import cv2, mss, torch
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

# ==================== Settings ====================

# Chrome window matching
OWNER_NAME = "Google Chrome"
TARGET_SUBSTR = ("fl511", "florida 511")

# YOLO model (nano = fastest)
YOLO_MODEL = "yolov5n"
CONF_THRES = 0.25
VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Crop to remove Chrome UI/black bars (toggle with 'c')
CROP = {"top": 110, "bottom": 30, "left": 140, "right": 140}

# Lane editor/persistence
LANE_FILE = Path("lanes.json")
LANE_ORDER = ["left", "mid", "right"]
LANE_COLORS = {"left": (255, 160, 40), "mid": (40, 220, 255), "right": (120, 255, 120)}
SELECT_RADIUS = 12  # px for grabbing a vertex

# Initial polygons (tweak roughly; then adjust on-screen)
LANES = {
    "left":  np.array([[120, 620], [520, 620], [560, 220], [160, 220]], dtype=np.int32),
    "mid":   np.array([[520, 620], [920, 620], [900, 220], [560, 220]], dtype=np.int32),
    "right": np.array([[920, 620], [1280, 620], [1240, 220], [900, 220]], dtype=np.int32),
}

# EMA smoothing for per-lane density
EMA_ALPHA = 0.30
ema_counts = {ln: 0.0 for ln in LANES}

# ==================== Helpers: Windows & Capture ====================

def list_windows():
    return CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []

def find_chrome_window_bounds_points():
    """Return (x,y,w,h,title) in mac POINTS for FL511 tab or largest Chrome window; else (None, '')."""
    wins = list_windows()
    # Prefer FL511 tab
    for w in wins:
        if w.get("kCGWindowOwnerName", "") == OWNER_NAME:
            title = (w.get("kCGWindowName") or "")
            if any(s in title.lower() for s in TARGET_SUBSTR):
                b = w.get("kCGWindowBounds", {})
                if b and b.get("Width", 0) > 50 and b.get("Height", 0) > 50:
                    return (int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"])), title
    # Fallback: largest Chrome window
    best = None; best_title = ""; area = -1
    for w in wins:
        if w.get("kCGWindowOwnerName", "") == OWNER_NAME:
            b = w.get("kCGWindowBounds", {})
            if b:
                a = int(b["Width"]) * int(b["Height"])
                if a > area:
                    area = a
                    best = (int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"]))
                    best_title = (w.get("kCGWindowName") or "")
    return (best, best_title) if best else (None, "")

def to_monitor_relative(sct, gx, gy, gw, gh):
    """Choose the MSS monitor containing the rect center; return relative rect + monitor info."""
    cx, cy = gx + gw // 2, gy + gh // 2
    mon_index, mon = None, None
    for i, m in enumerate(sct.monitors[1:], start=1):  # [0] is all monitors
        if m["left"] <= cx < m["left"] + m["width"] and m["top"] <= cy < m["top"] + m["height"]:
            mon_index, mon = i, m
            break
    if mon_index is None:  # fallback primary
        mon_index, mon = 1, sct.monitors[1]
    rel = {
        "mon": mon_index,
        "left": max(0, gx - mon["left"]),
        "top": max(0, gy - mon["top"]),
        "width": min(gw, mon["width"] - max(0, gx - mon["left"])),
        "height": min(gh, mon["height"] - max(0, gy - mon["top"])),
    }
    return rel, mon_index, mon

def is_blank(img):  # very dark/white frame detection
    return img.var() < 5.0

def safe_crop(img, crop):
    h, w = img.shape[:2]
    t = min(max(crop["top"], 0), h - 2)
    b = min(max(crop["bottom"], 0), h - 1 - t)
    l = min(max(crop["left"], 0), w - 2)
    r = min(max(crop["right"], 0), w - 1 - l)
    return img[t:h - b, l:w - r]

# ==================== Helpers: Lanes & Editing ====================

def save_lanes(lanes_dict):
    data = {k: v.astype(int).tolist() for k, v in lanes_dict.items()}
    LANE_FILE.write_text(json.dumps(data, indent=2))
    print(f"Saved {LANE_FILE.resolve()}")

def load_lanes():
    global LANES, ema_counts
    if LANE_FILE.exists():
        data = json.loads(LANE_FILE.read_text())
        LANES = {k: np.array(v, dtype=np.int32) for k, v in data.items()}
        # Reset EMA counts if lane set changed
        ema_counts = {ln: 0.0 for ln in LANES}
        print(f"Loaded {LANE_FILE.resolve()}")
    else:
        print("No lanes.json found; using defaults")

def box_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def count_per_lane(dets):
    counts = {ln: 0 for ln in LANES}
    membership = []
    for x1, y1, x2, y2, conf, cls in dets:
        cx, cy = box_center(x1, y1, x2, y2)
        placed = False
        for ln, poly in LANES.items():
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                counts[ln] += 1
                membership.append((ln, (x1, y1, x2, y2, conf, cls, cx, cy)))
                placed = True
                break
        if not placed:
            membership.append(("none", (x1, y1, x2, y2, conf, cls, cx, cy)))
    return counts, membership

def update_ema(raw_counts):
    for ln in LANES:
        ema_counts[ln] = EMA_ALPHA * raw_counts.get(ln, 0) + (1 - EMA_ALPHA) * ema_counts[ln]

def best_lane():
    return min(ema_counts.items(), key=lambda kv: kv[1])[0] if ema_counts else None

# --- interactive polygon editor ---
selected_lane = None
selected_idx = None
mouse_xy = (0, 0)

def nearest_vertex(poly, x, y):
    if poly is None or len(poly) == 0: return None
    dists = [(i, (p[0] - x) ** 2 + (p[1] - y) ** 2) for i, p in enumerate(poly)]
    i, d2 = min(dists, key=lambda t: t[1])
    return i if d2 <= SELECT_RADIUS ** 2 else None

def draw_lanes_edit(frame):
    # polygons + handles; highlight suggested lane
    sug = best_lane()
    if sug in LANES:
        cv2.polylines(frame, [LANES[sug]], True, (0, 255, 255), 4)
    for ln in LANE_ORDER:
        if ln not in LANES: continue
        poly = LANES[ln]
        color = LANE_COLORS.get(ln, (255, 255, 0))
        cv2.polylines(frame, [poly], True, color, 2)
        for i, (px, py) in enumerate(poly):
            fill = -1 if (ln == selected_lane and i == selected_idx) else 1
            cv2.circle(frame, (int(px), int(py)), 6, color, fill)

def mouse_cb(event, x, y, flags, param):
    global selected_lane, selected_idx, mouse_xy
    mouse_xy = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        lanes_to_search = [selected_lane] if selected_lane in LANES else LANE_ORDER
        for ln in lanes_to_search:
            idx = nearest_vertex(LANES[ln], x, y)
            if idx is not None:
                selected_lane, selected_idx = ln, idx
                break
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        if selected_lane in LANES and selected_idx is not None:
            LANES[selected_lane][selected_idx] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        pass

# ==================== YOLO ====================

print("Loading YOLO (torch hub)…")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = torch.hub.load("ultralytics/yolov5", YOLO_MODEL, trust_repo=True).to(device).eval()
names = model.names

def filter_dets(results):
    d = results.xyxy[0]
    if d is None or len(d) == 0: return []
    d = d.detach().cpu().numpy()
    out = []
    for x1, y1, x2, y2, conf, cls in d:
        cls = int(cls)
        if cls in VEHICLE_IDS and conf >= CONF_THRES:
            out.append((int(x1), int(y1), int(x2), int(y2), float(conf), cls))
    return out

# ==================== Main ====================

def main():
    global selected_lane, selected_idx

    last = time.time()
    fps = 0.0
    crop_on = True
    in_w = 640  # inference width (lower = faster)

    display_name = "Traffic Analytics (q quit | 1/2/3 lane | a add | d del | s save | l load | r reset | c crop | h help)"
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(display_name, mouse_cb)
    load_lanes()

    with mss.mss() as sct:
        while True:
            bounds_pts, title = find_chrome_window_bounds_points()
            if not bounds_pts:
                img = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(img, "Open Chrome FL511 tab…", (25, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(display_name, img)
                if cv2.waitKey(200) & 0xFF == ord('q'): break
                continue

            x_pt, y_pt, w_pt, h_pt = bounds_pts

            # Try Retina scale 2 then 1; choose first non-blank
            frame = None
            for SCALE in (2, 1):
                gx, gy, gw, gh = int(x_pt * SCALE), int(y_pt * SCALE), int(w_pt * SCALE), int(h_pt * SCALE)
                rel, mon_index, mon = to_monitor_relative(sct, gx, gy, gw, gh)
                img = np.array(sct.grab(rel))
                candidate = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                if not is_blank(candidate):
                    frame = candidate
                    break

            if frame is None:
                frame = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(frame, "Blank capture. Disable Chrome hardware acceleration & restart Chrome.",
                            (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(display_name, frame)
                if cv2.waitKey(200) & 0xFF == ord('q'): break
                continue

            if crop_on:
                frame = safe_crop(frame, CROP)

            # ---- Inference on downscaled copy (for speed) ----
            scale = frame.shape[1] / in_w
            small = cv2.resize(frame, (in_w, int(frame.shape[0] / scale)), interpolation=cv2.INTER_AREA)
            results = model(small, size=in_w)
            dets_small = results.xyxy[0].detach().cpu().numpy() if results.xyxy[0] is not None else []
            dets = []
            for x1, y1, x2, y2, conf, cls in dets_small:
                if int(cls) in VEHICLE_IDS and conf >= CONF_THRES:
                    dets.append((int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale), float(conf), int(cls)))

            # ---- Lane analytics ----
            lane_counts_raw, membership = count_per_lane(dets)
            update_ema(lane_counts_raw)

            # ---- Draw ----
            draw_lanes_edit(frame)
            # boxes colored by lane membership
            for ln, (x1, y1, x2, y2, conf, cls, cx, cy) in membership:
                color = LANE_COLORS.get(ln, (160, 160, 160)) if ln != "none" else (160, 160, 160)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)
                cv2.putText(frame, f"{names[cls]} {conf:.2f}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # HUD
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last))
            last = now
            cv2.rectangle(frame, (0, 0), (560, 22 * (6 + len(LANES))), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(frame, f"Win: {title[:48]}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Crop: {'ON' if crop_on else 'OFF'}", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            y = 84
            for ln in LANE_ORDER:
                cv2.putText(frame, f"{ln}: {lane_counts_raw.get(ln, 0)}  (EMA {ema_counts[ln]:.1f})",
                            (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LANE_COLORS.get(ln, (0, 255, 0)), 1)
                y += 22
            sug = best_lane()
            cv2.putText(frame, f"Suggested lane: {sug if sug else '—'}", (8, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            # mouse coords helper
            cv2.putText(frame, f"xy: {mouse_xy[0]},{mouse_xy[1]}",
                        (frame.shape[1] - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.rectangle(frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), (255, 255, 0), 2)

            cv2.imshow(display_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('1'), ord('2'), ord('3')):
                selected_lane = LANE_ORDER[int(chr(key)) - 1]
                print("Selected lane:", selected_lane)
            elif key == ord('a'):
                if selected_lane in LANES:
                    LANES[selected_lane] = np.append(LANES[selected_lane], [mouse_xy], axis=0)
                    selected_idx = len(LANES[selected_lane]) - 1
                    print(f"Added vertex to {selected_lane} at {mouse_xy}")
            elif key == ord('d'):
                if selected_lane in LANES and selected_idx is not None and len(LANES[selected_lane]) > 3:
                    LANES[selected_lane] = np.delete(LANES[selected_lane], selected_idx, axis=0)
                    selected_idx = None
                    print(f"Deleted vertex from {selected_lane}")
            elif key == ord('s'):
                save_lanes(LANES)
            elif key == ord('l'):
                load_lanes()
            elif key == ord('r'):
                for ln in ema_counts: ema_counts[ln] = 0.0
                print("EMA reset")
            elif key == ord('c'):
                crop_on = not crop_on
            elif key == ord('h'):
                print("[q] quit | [1/2/3] select lane | [drag] move vertex | [a] add | [d] delete | [s] save | [l] load | [r] reset EMA | [c] toggle crop")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
