import cv2, mediapipe as mp, numpy as np, tensorflow as tf, pickle, time
from collections import deque
import pyttsx3
from utils_preproc_two import pack_two_hands, smooth_labels

MODEL_PATH = "model/isl_model_two.keras"
ENCODER_PATH = "model/label_encoder_two.pkl"
THRESHOLD = 0.60
SMOOTH_WINDOW = 12
SPEAK_COOLDOWN = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_text(img, text, org, scale=0.9, color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def main():
    # Load model and encoder
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f: 
        le = pickle.load(f)
    idx2label = {i:l for i,l in enumerate(le.classes_)}

    # TTS engine
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    last_spoken = ""
    last_t = 0.0

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam 0.")

    # ✅ FIX: Use keyword arguments here
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    preds = deque(maxlen=SMOOTH_WINDOW)
    fps_hist = deque(maxlen=30)
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: 
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h,w = frame.shape[:2]
        label_show = ""
        conf_show = 0.0

        hands_dict = {'Left': None, 'Right': None}
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm,handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                lr = handed.classification[0].label
                pts = np.array([[p.x,p.y,p.z] for p in lm.landmark], dtype=np.float32)
                hands_dict[lr] = pts
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        feats, mask = pack_two_hands(hands_dict)
        inp = np.concatenate([feats, mask], axis=0)[None,:]
        probs = model.predict(inp, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(np.max(probs))
        label = idx2label[pred_idx]

        if conf >= THRESHOLD:
            preds.append(label)
            label_show = label
            conf_show = conf
        else:
            preds.append("")
            label_show = ""
            conf_show = conf

        stable = smooth_labels(list(preds), min_count=SMOOTH_WINDOW//2 + 1)
        if stable: 
            label_show = stable

        now = time.time()
        if label_show and (label_show != last_spoken or now - last_t > SPEAK_COOLDOWN):
            engine.stop()
            engine.say(label_show.replace("_"," "))
            engine.runAndWait()
            last_spoken = label_show
            last_t = now

        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        fps_hist.append(fps)
        fps_avg = sum(fps_hist)/len(fps_hist)

        draw_text(frame, "ISL (Two Hands): Real-time Recognition", (10,30))
        draw_text(frame, f"FPS: {fps_avg:5.1f}", (10,60), 0.8)
        draw_text(frame, f"Conf: {conf_show*100:5.1f}%", (10,90), 0.8)
        if label_show:
            cv2.rectangle(frame,(w//2-250,h-110),(w//2+250,h-40),(0,255,0),-1)
            draw_text(frame, label_show.upper(), (w//2-230, h-60), 1.0, (0,0,0))
        else:
            draw_text(frame, "Show a known two-hand sign...", (w//2-230, h-60), 0.8, (0,255,255))

        cv2.imshow("ISL — Two Hands (Text + Speech)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
