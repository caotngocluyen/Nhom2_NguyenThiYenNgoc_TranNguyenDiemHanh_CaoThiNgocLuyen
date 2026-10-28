"""Patch src/08_train_hybrid.py for efficiency and correctness."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/08_train_hybrid.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update fit() method signature and implementation
old_fit = '''    def fit(self, svd_model, cb_model, df_train):
        """
        Thiết lập Hybrid model từ các mô hình thành phần đã huấn luyện.
        
        Tham số:
            svd_model: Surprise SVD model đã fit
            cb_model: ContentBasedRecommender đã fit
            df_train: Tập huấn luyện để xây dựng user histories
        """
        self.cf_model = svd_model
        self.cb_model = cb_model
        self.global_mean = df_train["rating"].mean()

        # Xây dựng user histories
        logger.info("Xây dựng user histories từ train")
        self.user_histories = df_train.groupby("userId").apply(
            lambda x: dict(zip(x["movieId"], x["rating"]))
        ).to_dict()

        logger.info(f"  Số user có history: {len(self.user_histories):,}")
        return self'''

new_fit = '''    def fit(self, svd_model, cb_model, df_train, active_users=None):
        """
        Thiết lập Hybrid model từ các mô hình thành phần đã huấn luyện.
        
        Tham số:
            svd_model: Surprise SVD model đã fit
            cb_model: ContentBasedRecommender đã fit
            df_train: Tập huấn luyện để xây dựng user histories
            active_users: Tập các userId cần xây dựng history (giúp tiết kiệm RAM & thời gian)
        """
        self.cf_model = svd_model
        self.cb_model = cb_model
        self.global_mean = float(df_train["rating"].mean())

        # Xây dựng user histories cho active users (hoặc toàn bộ nếu không chỉ định)
        logger.info("Xây dựng user histories từ train")
        if active_users is not None:
            df_sub = df_train[df_train["userId"].isin(active_users)]
            logger.info(f"  Lọc {len(active_users):,} active users: {len(df_sub):,} ratings")
        else:
            df_sub = df_train

        self.user_histories = df_sub.groupby("userId").apply(
            lambda x: dict(zip(x["movieId"], x["rating"]))
        ).to_dict()

        logger.info(f"  Số user có history: {len(self.user_histories):,}")
        return self'''

if old_fit in content:
    content = content.replace(old_fit, new_fit)
    print("Updated fit()")
else:
    print("old_fit not found!")

# 2. Fix all_cf_items in recommend_for_user
old_cf_items = '''        all_cf_items = set(
            int(iid) for iid in self.cf_model.trainset.all_items()
        ) - seen_movies'''

new_cf_items = '''        all_cf_items = set(
            int(self.cf_model.trainset.to_raw_iid(inner_id))
            for inner_id in self.cf_model.trainset.all_items()
        ) - seen_movies'''

if old_cf_items in content:
    content = content.replace(old_cf_items, new_cf_items)
    print("Updated all_cf_items")
else:
    print("old_cf_items not found!")

# 3. Update save() to avoid saving full 20M user_histories dict
old_save = '''    def save(self, path):
        # Lưu không lưu cf_model và cb_model (đã lưu riêng)
        state = {
            "alpha": self.alpha,
            "strategy": self.strategy,
            "global_mean": self.global_mean,
            "user_histories": self.user_histories,
        }
        joblib.dump(state, path)
        logger.info(f"Đã lưu Hybrid model state: {path}")'''

new_save = '''    def save(self, path):
        # Lưu cấu hình Hybrid (cf_model và cb_model đã được lưu riêng)
        state = {
            "alpha": self.alpha,
            "strategy": self.strategy,
            "global_mean": self.global_mean,
        }
        joblib.dump(state, path)
        logger.info(f"Đã lưu Hybrid model config: {path}")'''

if old_save in content:
    content = content.replace(old_save, new_save)
    print("Updated save()")
else:
    print("old_save not found!")

# 4. In main(), pass active_users to fit()
old_call = '''    hybrid = HybridRecommender(alpha=0.6, strategy="weighted")
    hybrid.fit(svd_model, cb_model, df_train)'''

new_call = '''    hybrid = HybridRecommender(alpha=0.6, strategy="weighted")
    # Thu thập các user cần dùng history: tập val sample + top active users cho demo
    demo_users = df_train.groupby("userId").size().nlargest(5).index.tolist()
    active_users = set(df_val_sample["userId"]).union(demo_users)
    hybrid.fit(svd_model, cb_model, df_train, active_users=active_users)'''

if old_call in content:
    content = content.replace(old_call, new_call)
    print("Updated hybrid.fit call")
else:
    print("old_call not found!")

# 5. In main() demo, use demo_users
old_demo = '''    sample_users = list(hybrid.user_histories.keys())[:3]'''
new_demo = '''    sample_users = demo_users[:3]'''
if old_demo in content:
    content = content.replace(old_demo, new_demo)
    print("Updated sample_users")
else:
    print("old_demo not found!")

with open('src/08_train_hybrid.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied to src/08_train_hybrid.py")
