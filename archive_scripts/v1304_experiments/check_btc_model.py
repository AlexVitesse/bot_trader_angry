import joblib
import sys
print("Loading model...", flush=True)
sys.stdout.flush()
m = joblib.load('models/v95_v7_BTCUSDT.pkl')
print(f"Features count: {len(m.feature_name_)}", flush=True)
print(f"First 10: {m.feature_name_[:10]}", flush=True)
