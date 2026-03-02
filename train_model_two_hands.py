# train_model_two_hands.py  — improved version
import os, pickle, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_CSV = "data/isl_landmarks_two.csv"
MODEL_PATH = "model/isl_model_two.keras"
ENCODER_PATH = "model/label_encoder_two.pkl"

os.makedirs("model", exist_ok=True)

def build_mlp(input_dim, num_classes):
    """Enhanced MLP model with BatchNorm, Dropout, and deeper layers"""
    i = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(512, activation='relu')(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    o = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    m = tf.keras.Model(i, o)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found. Run collect_data_two_hands.py first.")
    df = pd.read_csv(DATA_CSV)
    feat = [c for c in df.columns if c.startswith("f")]
    mask = ["has_left","has_right"]
    available_cols = [c for c in feat + ["has_left", "has_right"] if c in df.columns]
    X = df[available_cols].values.astype('float32')

    y_txt = df["label"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_txt)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    m = build_mlp(X.shape[1], len(le.classes_))

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', patience=6, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(
            patience=12, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor="val_accuracy")
    ]

    hist = m.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=150,
        batch_size=32,
        verbose=2,
        callbacks=cbs
    )

    m.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    _, acc = m.evaluate(Xva, yva, verbose=0)
    print("✅ Classes:", list(le.classes_))
    print(f"✅ Validation accuracy: {acc*100:.2f}%")

    # Optionally save history
    pd.DataFrame(hist.history).to_csv("model/training_history.csv", index=False)
    print("Training history saved to model/training_history.csv")

if __name__ == "__main__":
    main()
