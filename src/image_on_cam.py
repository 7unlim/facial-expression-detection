import cv2, numpy as np

CAMERA_INDEX = 1
OVERLAY_PATH = "datboy-removebg-preview.png"

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError(f"Camera {CAMERA_INDEX} failed to open.")

overlay = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)  # RGB[A]
if overlay is None:
    raise FileNotFoundError(OVERLAY_PATH)

# If the image has no alpha channel, add a fully opaque one
if overlay.ndim == 3 and overlay.shape[2] == 3:
    h, w = overlay.shape[:2]
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    overlay = np.concatenate([overlay, alpha], axis=2)  # now BGRA

# Pre-resize once (adjust scale)
scale = 0.5
overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

ov_bgr   = overlay[:, :, :3].astype(np.float32)
ov_alpha = (overlay[:, :, 3:4].astype(np.float32) / 255.0)  # (h,w,1)
oh, ow   = overlay.shape[:2]

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    x = W - ow - 12
    y = H - oh - 12
    # if x < 0 or y < 0:
    #     # overlay bigger than frame; resize smaller once and continue
    #     scale *= 0.5
    #     overlay = cv2.resize(overlay, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    #     ov_bgr   = overlay[:, :, :3].astype(np.float32)
    #     ov_alpha = (overlay[:, :, 3:4].astype(np.float32) / 255.0)
    #     oh, ow   = overlay.shape[:2]
    #     continue

    roi = frame[y:y+oh, x:x+ow].astype(np.float32)
    blended = ov_bgr * ov_alpha + roi * (1.0 - ov_alpha)
    frame[y:y+oh, x:x+ow] = blended.astype(np.uint8)

    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
