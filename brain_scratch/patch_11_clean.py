import os

path = 'src/11_recommend.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. SVD model preference
text = text.replace(
    'for svd_name in ["best_svd_model.pkl", "svd_model.pkl"]:',
    'for svd_name in ["svd_improved.pkl", "best_svd_model.pkl", "svd_model.pkl"]:'
)

# 2. Integer predict
text = text.replace(
    'preds = [(m, svd.predict(str(user_id), str(m)).est) for m in sample_items]',
    'preds = [(m, svd.predict(int(user_id), int(m)).est) for m in sample_items]'
)
text = text.replace(
    'cf_p = svd.predict(str(user_id), str(mid)).est',
    'try:\n                cf_p = svd.predict(int(user_id), int(mid)).est\n            except:\n                cf_p = svd.predict(str(user_id), str(mid)).est'
)

# 3. Train improved file
text = text.replace(
    'df_train = pd.read_parquet(config.TRAIN_FILE)',
    'train_f = os.path.join(config.DATA_PROCESSED_DIR, "train_improved.parquet") if os.path.exists(os.path.join(config.DATA_PROCESSED_DIR, "train_improved.parquet")) else config.TRAIN_FILE\n    df_train = pd.read_parquet(train_f)'
)

# 4. groupby
text = text.replace(
    'lambda x: dict(zip(x["movieId"], x["rating"]))',
    'lambda x: dict(zip(x["movieId"], x["rating"])), include_groups=False'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully updated src/11_recommend.py")
