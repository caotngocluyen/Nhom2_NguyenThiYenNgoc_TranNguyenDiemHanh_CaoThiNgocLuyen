"""Patch src/10_evaluate.py to:
1. Optimize user_histories construction by filtering to evaluated users.
2. Fix hybrid rating combination to direct linear combination on star scale.
3. Fix hybrid candidate pool generation for fast and accurate top-N evaluation.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/10_evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Patch user_histories building in danh_gia_tat_ca
old_hist = '''    # Xây dựng user histories
    logger.info("Xây dựng user histories")
    user_histories = df_train.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"]))
    ).to_dict()
    global_mean = df_train["rating"].mean()'''

new_hist = '''    # Xây dựng user histories cho các user cần đánh giá trong val và test (tối ưu tốc độ)
    logger.info("Xây dựng user histories cho các user đánh giá")
    eval_user_ids = set(df_val.sample(min(30000, len(df_val)), random_state=config.RANDOM_SEED)["userId"]).union(
        set(df_test.sample(min(30000, len(df_test)), random_state=config.RANDOM_SEED)["userId"])
    )
    df_train_sub = df_train[df_train["userId"].isin(eval_user_ids)]
    user_histories = df_train_sub.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"]))
    ).to_dict()
    global_mean = float(df_train["rating"].mean())
    logger.info(f"  Đã tải user histories cho {len(user_histories):,} users đánh giá.")'''

if old_hist in content:
    content = content.replace(old_hist, new_hist)
    print("Patched user_histories in 10_evaluate.py")
else:
    print("old_hist not found")

# 2. Patch hybrid_predict and hybrid_recommend in 10_evaluate.py
old_hybrid_block = '''        def hybrid_predict(user_id, movie_id):
            cf_p = svd.predict(str(user_id), str(movie_id)).est
            cb_p = cb.predict_rating(user_histories.get(user_id, {}), movie_id)
            # Normalize
            cf_n = (cf_p - config.MIN_RATING) / (config.MAX_RATING - config.MIN_RATING)
            cb_n = max(0, min(1, cb_p))
            hybrid = alpha * cf_n + (1 - alpha) * cb_n
            return hybrid * (config.MAX_RATING - config.MIN_RATING) + config.MIN_RATING'''

new_hybrid_block = '''        def hybrid_predict(user_id, movie_id):
            cf_p = svd.predict(str(user_id), str(movie_id)).est
            cb_p = cb.predict_rating(user_histories.get(user_id, {}), movie_id)
            hybrid = alpha * cf_p + (1.0 - alpha) * cb_p
            return float(np.clip(hybrid, config.MIN_RATING, config.MAX_RATING))'''

if old_hybrid_block in content:
    content = content.replace(old_hybrid_block, new_hybrid_block)
    print("Patched hybrid_predict in 10_evaluate.py")
else:
    print("old_hybrid_block not found")

# 3. Patch hybrid_recommend candidate generation in 10_evaluate.py
old_hybrid_rec = '''        # Top-N Hybrid
        def hybrid_recommend(user_id, seen):
            candidates = [m for m in all_train_movies if m not in seen]
            np.random.shuffle(candidates)
            candidates = candidates[:2000]

            scores = []
            history = user_histories.get(user_id, {})
            for mid in candidates:
                cf_p = svd.predict(str(user_id), str(mid)).est
                cb_p = cb.predict_rating(history, mid)
                cf_n = (cf_p - config.MIN_RATING) / (config.MAX_RATING - config.MIN_RATING)
                cb_n = max(0, min(1, cb_p))
                h = alpha * cf_n + (1 - alpha) * cb_n
                scores.append((mid, h))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in scores[:config.TOP_K]]'''

new_hybrid_rec = '''        # Top-N Hybrid (Cascade Candidate Generation)
        def hybrid_recommend(user_id, seen):
            history = user_histories.get(user_id, {})
            # Lấy top ứng viên từ CF và CB
            cf_candidates = [m for m in all_train_movies if m not in seen]
            np.random.shuffle(cf_candidates)
            pool = cf_candidates[:200]
            if history:
                cb_recs = cb.recommend_for_user(history, n=50, exclude_seen=True)
                pool = list(set(pool).union(m for m, _ in cb_recs))

            scores = []
            for mid in pool:
                cf_p = svd.predict(str(user_id), str(mid)).est
                cb_p = cb.predict_rating(history, mid)
                h = alpha * cf_p + (1.0 - alpha) * cb_p
                scores.append((mid, h))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in scores[:config.TOP_K]]'''

if old_hybrid_rec in content:
    content = content.replace(old_hybrid_rec, new_hybrid_rec)
    print("Patched hybrid_recommend in 10_evaluate.py")
else:
    print("old_hybrid_rec not found")

with open('src/10_evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Saved patched src/10_evaluate.py")
