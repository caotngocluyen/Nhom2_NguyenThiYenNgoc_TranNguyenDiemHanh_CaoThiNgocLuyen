"""Patch 09_tune_models.py to:
1. Optimize user_histories in tune_hybrid_alpha (filter to val_sample users)
2. Fix hybrid prediction combination (linear combination of raw star predictions)
3. Set n_trials = 3 for SVD random search (so it finishes in reasonable time)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/09_tune_models.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Patch n_trials in tune_svd
old_trials = 'n_trials = min(8, len(all_combinations))  # Tối đa 8 lần thử'
new_trials = 'n_trials = min(3, len(all_combinations))  # 3 lần thử có kiểm soát để hoàn thành trong 15-20 phút'
if old_trials in content:
    content = content.replace(old_trials, new_trials)
    print("Patched n_trials")
else:
    print("old_trials not found")

# 2. Patch tune_hybrid_alpha
old_tune_hybrid_start = '''    # Xây dựng user histories
    logger.info("Xây dựng user histories")
    user_histories = df_train.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"]))
    ).to_dict()
    global_mean = df_train["rating"].mean()

    # Lấy mẫu val để đánh giá nhanh
    n_sample = min(3000, len(df_val))
    df_val_sample = df_val.sample(n_sample, random_state=config.RANDOM_SEED)
    y_true = df_val_sample["rating"].values.astype(np.float32)'''

new_tune_hybrid_start = '''    # Lấy mẫu val để đánh giá nhanh
    n_sample = min(3000, len(df_val))
    df_val_sample = df_val.sample(n_sample, random_state=config.RANDOM_SEED)
    y_true = df_val_sample["rating"].values.astype(np.float32)

    # Xây dựng user histories chỉ cho các user có trong mẫu val (tối ưu tốc độ & RAM)
    logger.info("Xây dựng user histories cho val sample")
    val_sample_users = set(df_val_sample["userId"])
    df_train_sub = df_train[df_train["userId"].isin(val_sample_users)]
    user_histories = df_train_sub.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"]))
    ).to_dict()
    global_mean = float(df_train["rating"].mean())
    logger.info(f"  Số user có history trong val sample: {len(user_histories):,}")'''

if old_tune_hybrid_start in content:
    content = content.replace(old_tune_hybrid_start, new_tune_hybrid_start)
    print("Patched tune_hybrid_alpha user_histories")
else:
    print("old_tune_hybrid_start not found")

# 3. Patch combination logic in tune_hybrid_alpha
old_comb = '''    # Normalize về [0, 1]
    def norm_minmax(x):
        mn, mx = x.min(), x.max()
        if mx == mn: return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    cf_norm = norm_minmax(cf_preds)
    cb_norm = np.clip(cb_preds, 0, 1)  # CB đã gần [0,1]

    # Thử từng alpha
    alpha_results = []
    logger.info(f"\\nThử {len(config.HYBRID_ALPHA_RANGE)} giá trị alpha")

    for alpha in config.HYBRID_ALPHA_RANGE:
        # Tính hybrid predictions
        hybrid_norm = alpha * cf_norm + (1 - alpha) * cb_norm
        hybrid_preds = hybrid_norm * (config.MAX_RATING - config.MIN_RATING) + config.MIN_RATING
        hybrid_preds = np.clip(hybrid_preds, config.MIN_RATING, config.MAX_RATING)
        
        rmse = tinh_rmse(y_true, hybrid_preds)
        mae  = float(np.mean(np.abs(y_true - hybrid_preds)))
        
        alpha_results.append({
            "model_type": "Hybrid_Alpha",
            "alpha": alpha,
            "rmse_val": rmse,
            "mae_val": mae,
        })
        logger.info(f"  alpha={alpha:.1f}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Tìm alpha tốt nhất
    best_result = min(alpha_results, key=lambda x: x["rmse_val"])
    best_alpha = best_result["alpha"]
    best_rmse  = best_result["rmse_val"]

    logger.info(f"\\n Alpha tốt nhất: {best_alpha} (RMSE={best_rmse:.4f})")
    logger.info(f"  So sánh:")
    logger.info(f"    Thuần CF (alpha=1.0):  RMSE={tinh_rmse(y_true, (cf_norm*(config.MAX_RATING-config.MIN_RATING)+config.MIN_RATING)):.4f}")
    logger.info(f"    Thuần CB (alpha=0.0):  RMSE={tinh_rmse(y_true, (cb_norm*(config.MAX_RATING-config.MIN_RATING)+config.MIN_RATING)):.4f}")
    logger.info(f"    Best Hybrid:           RMSE={best_rmse:.4f}")'''

new_comb = '''    # Cả CF và CB đều dự đoán trên thang điểm rating [0.5, 5.0]
    # Kết hợp tuyến tính trực tiếp theo trọng số alpha
    alpha_results = []
    logger.info(f"\\nThử {len(config.HYBRID_ALPHA_RANGE)} giá trị alpha")

    for alpha in config.HYBRID_ALPHA_RANGE:
        hybrid_preds = alpha * cf_preds + (1.0 - alpha) * cb_preds
        hybrid_preds = np.clip(hybrid_preds, config.MIN_RATING, config.MAX_RATING)
        
        rmse = tinh_rmse(y_true, hybrid_preds)
        mae  = float(np.mean(np.abs(y_true - hybrid_preds)))
        
        alpha_results.append({
            "model_type": "Hybrid_Alpha",
            "alpha": alpha,
            "rmse_val": rmse,
            "mae_val": mae,
        })
        logger.info(f"  alpha={alpha:.1f}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Tìm alpha tốt nhất
    best_result = min(alpha_results, key=lambda x: x["rmse_val"])
    best_alpha = best_result["alpha"]
    best_rmse  = best_result["rmse_val"]

    logger.info(f"\\n Alpha tốt nhất: {best_alpha} (RMSE={best_rmse:.4f})")
    logger.info(f"  So sánh:")
    logger.info(f"    Thuần CF (alpha=1.0):  RMSE={tinh_rmse(y_true, cf_preds):.4f}")
    logger.info(f"    Thuần CB (alpha=0.0):  RMSE={tinh_rmse(y_true, cb_preds):.4f}")
    logger.info(f"    Best Hybrid:           RMSE={best_rmse:.4f}")'''

if old_comb in content:
    content = content.replace(old_comb, new_comb)
    print("Patched combination logic")
else:
    print("old_comb not found")

with open('src/09_tune_models.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Saved patched src/09_tune_models.py")
