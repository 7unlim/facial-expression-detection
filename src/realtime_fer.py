import cv2
import numpy as np
from fer import FER

CAMERA_INDEX = 1
OVERLAY_PATH = "../assets/datboy-removebg-preview.png"
HAPPY_THRESH = 0.60 # confidence needed to trigger
SCALE_FACTOR = 1.1 # overlay width relative to face width
MARGIN = 8 # pixels above the face box

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError(f"Camera {CAMERA_INDEX} failed to open.")

# Load overlay (ensure alpha exists)
overlay = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)  # BGR[A]
if overlay is None:
    raise FileNotFoundError(OVERLAY_PATH)
if overlay.ndim == 3 or overlay.shape[2] == 3:
    h, w = overlay.shape[:2]
    overlay = np.dstack([overlay, np.full((h, w, 1), 255, np.uint8)])

detector = FER(mtcnn=False)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    H, W = frame.shape[:2]

    results = detector.detect_emotions(frame)
    for r in results:
        fx, fy, fw, fh = r["box"]
        emo = r["emotions"]
        label = max(emo, key=emo.get); score = emo[label]

        # draw face box + label
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 160, 255), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (fx, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 255), 2)

        if label == "happy" and score >= HAPPY_THRESH:
            # place overlay centered above the face 
            oh0, ow0 = overlay.shape[:2]
            target_w = max(1, int(fw * SCALE_FACTOR))
            target_h = max(1, int(oh0 * (target_w / ow0)))
            ov = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

            ov_bgr   = ov[:, :, :3].astype(np.float32)
            ov_alpha = (ov[:, :, 3:4].astype(np.float32) / 255.0)  # (h,w,1)
            oh, ow   = ov.shape[:2]

            ox = int(fx + fw/2 - ow/2)
            oy = int(fy - oh - MARGIN)

            # # crop if overlay goes off-frame
            # x1, y1 = max(0, ox), max(0, oy)
            # x2, y2 = min(W, ox + ow), min(H, oy + oh)
            # if x1 < x2 and y1 < y2:
            #     ox1, oy1 = x1 - ox, y1 - oy
            #     ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

            #     roi = frame[y1:y2, x1:x2].astype(np.float32)
            #     ov_rgb_c = ov_bgr[oy1:oy2, ox1:ox2]
            #     ov_a_c   = ov_alpha[oy1:oy2, ox1:ox2]

            #     blended = ov_rgb_c * ov_a_c + roi * (1.0 - ov_a_c)
            #     frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    cv2.imshow("Facial Expression", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
