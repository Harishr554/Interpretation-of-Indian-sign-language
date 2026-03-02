import pandas as pd

# Path to your dataset
DATA_CSV = "data/isl_landmarks_two.csv"

# The gesture labels you want to delete (case-sensitive)
GESTURES_TO_DELETE = ["NAMASTE", "INDIAN"]

# Load dataset
df = pd.read_csv(DATA_CSV)
print(f"Original dataset size: {len(df)} rows")

# Count before deletion
for gesture in GESTURES_TO_DELETE:
    count = len(df[df['label'] == gesture])
    print(f"Found {count} samples of '{gesture}'")

# Filter out the unwanted gestures
df_new = df[~df['label'].isin(GESTURES_TO_DELETE)]
print(f"After deleting {GESTURES_TO_DELETE}: {len(df_new)} rows (removed {len(df) - len(df_new)} rows)")

# Save it back (overwrite)
df_new.to_csv(DATA_CSV, index=False)
print(f"✅ Deleted all '{GESTURES_TO_DELETE}' samples successfully.")
