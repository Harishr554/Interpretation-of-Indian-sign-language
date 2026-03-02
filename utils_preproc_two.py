
import numpy as np

def normalize_landmarks_single(hand_landmarks):
    lm = np.array(hand_landmarks, dtype=np.float32)
    wrist = lm[0].copy()
    lm -= wrist
    d = np.linalg.norm(lm[:, :2], axis=1)
    s = np.max(d)
    if s < 1e-6: s = 1.0
    lm[:, :3] /= s
    return lm.flatten().astype(np.float32)

def pack_two_hands(hands_dict):
    feat_left = np.zeros((63,), dtype=np.float32)
    feat_right = np.zeros((63,), dtype=np.float32)
    has_left = 0.0; has_right = 0.0
    if hands_dict.get('Left') is not None:
        feat_left = normalize_landmarks_single(hands_dict['Left']); has_left = 1.0
    if hands_dict.get('Right') is not None:
        feat_right = normalize_landmarks_single(hands_dict['Right']); has_right = 1.0
    features = np.concatenate([feat_left, feat_right], axis=0)
    mask = np.array([has_left, has_right], dtype=np.float32)
    return features, mask

def smooth_labels(buffer, min_count=5):
    if not buffer: return None
    vals, counts = np.unique(buffer, return_counts=True)
    idx = np.argmax(counts)
    if counts[idx] >= min_count and vals[idx] != "":
        return vals[idx]
    return None
