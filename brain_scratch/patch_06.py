"""Patch 06_train_cf.py to sample val for evaluation instead of looping all 2.6M rows."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/06_train_cf.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = 'def danh_gia_tren_val(mo_hinh, df_val, reader):'
end_marker = '\ndef luu_du_lieu_svd_cho_hybrid'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1:
    print("ERROR: start_marker not found!")
    exit(1)
if end_idx == -1:
    print("ERROR: end_marker not found!")
    exit(1)

print(f"Found function at chars {start_idx}:{end_idx}")

new_func = '''def danh_gia_tren_val(mo_hinh, df_val, reader):
    """
    Danh gia mo hinh SVD tren tap validation.

    De tranh lap qua 2.6M dong, ta lay mau 10,000 cap
    de danh gia nhanh va du chinh xac thong ke.

    surprise.predict(uid, iid) tra ve Prediction object:
    - est: rating du doan
    - details.was_impossible: True neu cold-start
    """
    logger.info("\\nDu doan va danh gia tren tap validation")

    # Lay mau de danh gia nhanh (10,000 cap)
    n_eval = min(10000, len(df_val))
    df_eval = df_val.sample(n_eval, random_state=42)
    logger.info(f"  Danh gia tren {n_eval:,} mau ngau nhien tu {len(df_val):,} ratings val")

    y_true = []
    y_pred = []
    n_cold = 0

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Danh gia val"):
        uid = str(int(row["userId"]))
        iid = str(int(row["movieId"]))
        actual = float(row["rating"])

        pred = mo_hinh.predict(uid, iid)
        y_true.append(actual)
        y_pred.append(pred.est)

        if pred.details.get("was_impossible"):
            n_cold += 1

    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))

    logger.info(f"\\n Ket qua SVD tren Validation (sample {n_eval:,}) ")
    logger.info(f"  RMSE:       {rmse:.4f}")
    logger.info(f"  MAE:        {mae:.4f}")
    logger.info(f"  Cold-start: {n_cold:,} ({n_cold/n_eval*100:.2f}%)")

    return rmse, mae, y_pred, n_cold'''

new_content = content[:start_idx] + new_func + content[end_idx:]

with open('src/06_train_cf.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Patch applied successfully!")
