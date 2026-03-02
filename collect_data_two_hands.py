import os, cv2, mediapipe as mp, numpy as np, pandas as pd
from datetime import datetime
from utils_preproc_two import pack_two_hands

CLASSES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9","Yes","NO","THANKYOU","I AM","Namaste","Indian"]
OUT_CSV = "data/isl_landmarks_two.csv"
SAMPLES_PER_CLASS = 180
SHOW_SKELETON = True

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_text(img, text, org, scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Could not open webcam 0.")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)

    cols = [f"f{i}" for i in range(126)] + ["has_left","has_right","label","ts"]
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(OUT_CSV):
        pd.DataFrame(columns=cols).to_csv(OUT_CSV, index=False)

    idx = 0; collected = 0; target = CLASSES[idx]; recording = False
    print("SPACE: record | n: next | p: prev | q: quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h, w = frame.shape[:2]
        hands_dict = {'Left': None, 'Right': None}
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                lr = handed.classification[0].label
                pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                hands_dict[lr] = pts
                if SHOW_SKELETON:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                              mp_styles.get_default_hand_landmarks_style(),
                                              mp_styles.get_default_hand_connections_style())

        feats, mask = pack_two_hands(hands_dict)
        if recording:
            row = list(feats) + list(mask) + [target, datetime.utcnow().isoformat()]
            pd.DataFrame([row], columns=cols).to_csv(OUT_CSV, mode="a", header=False, index=False)
            collected += 1

        draw_text(frame, f"CLASS: {target} ({idx+1}/{len(CLASSES)})", (10,30))
        draw_text(frame, f"Collected: {collected}/{SAMPLES_PER_CLASS}", (10,60))
        draw_text(frame, "SPACE: record | n: next | p: prev | q: quit", (10, h-20), 0.6)
        if recording:
            cv2.rectangle(frame,(0,0),(w,80),(0,165,255),-1)
            draw_text(frame,"RECORDING (two hands)...",(10,55),0.9,(0,0,0))

        cv2.imshow("Collect ISL Data (Two Hands)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            recording = not recording
            if recording: collected = 0
            else: print(f"Saved ~{collected} frames for '{target}'.")
        elif k == ord('n'):
            idx = (idx+1)%len(CLASSES); target = CLASSES[idx]; collected = 0; recording = False
        elif k == ord('p'):
            idx = (idx-1)%len(CLASSES); target = CLASSES[idx]; collected = 0; recording = False
        elif k == ord('q'): break
        if recording and collected >= SAMPLES_PER_CLASS:
            recording = False; print(f"Auto-stopped at {SAMPLES_PER_CLASS} for '{target}'.")

    hands.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()