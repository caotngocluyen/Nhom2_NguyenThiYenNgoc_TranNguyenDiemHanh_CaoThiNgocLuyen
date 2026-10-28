import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.two_stage_recommender import TwoStageRecommender
from src.semantic_embeddings import SemanticContentRecommender

log_file = os.path.join(config.REPORTS_DIR, "08_reranker_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def train_two_stage_reranker(n_sample_users=5000, n_pos_per_user=5, n_neg_per_user=5):
    """
    Huan luyen mo hinh LightGBM Re-ranker voi ham muc tieu LambdaRank (Xep hang danh sach)
    va bo sung dac trung Trùng khớp the loai (Genre Affinity Feature).
    
    Tham so:
        n_sample_users: So luong nguoi dung lay mau de huan luyen.
        n_pos_per_user: So mau duong cho moi nguoi dung.
        n_neg_per_user: So mau am (negative) cho moi nguoi dung.
    """
    logger.info("Bat dau quy trinh huan luyen LightGBM Re-ranker (LambdaRank + Genre Affinity)")

    train_path = config.TRAIN_IMPROVED_FILE
    logger.info("Dang doc du lieu tu: %s", train_path)
    df_train = pd.read_parquet(train_path, columns=["userId", "movieId", "rating"])

    # 1. Doc the loai phim de phuc vu dac trung Genre Affinity
    logger.info("Doc thong tin the loai phim tu: %s", config.MOVIES_CLEAN_FILE)
    df_movies = pd.read_parquet(config.MOVIES_CLEAN_FILE, columns=["movieId", "genres_text"])
    movie_genres = {
        int(m): set(g.split())
        for m, g in zip(df_movies["movieId"], df_movies["genres_text"].fillna(""))
        if g
    }
    logger.info("Da nap the loai cho %d bo phim", len(movie_genres))

    # 2. Tinh toan thong ke cho nguoi dung va bo phim
    logger.info("Tinh thong ke danh gia cho phim va nguoi dung")
    movie_group = df_train.groupby("movieId")["rating"].agg(["mean", "count"])
    movie_stats = {
        int(m): {"avg_rating": float(row["mean"]), "count": int(row["count"])}
        for m, row in movie_group.iterrows()
    }

    user_group = df_train.groupby("userId")["rating"].agg(["mean", "count"])
    user_stats = {
        int(u): {"avg_rating": float(row["mean"]), "count": int(row["count"])}
        for u, row in user_group.iterrows()
    }

    popular_movies = list(movie_group.sort_values(by="count", ascending=False).index[:500])
    all_movie_ids = list(movie_group.index)

    # 3. Nap mo hinh SVD va Semantic Embeddings
    logger.info("Nap mo hinh SVD va Semantic Embeddings")
    svd = joblib.load(config.SVD_IMPROVED_MODEL)
    semantic = SemanticContentRecommender.load(config.MODELS_DIR)

    pipeline = TwoStageRecommender(svd_model=svd, semantic_model=semantic)
    pipeline.set_stats(movie_stats, user_stats, popular_movies, all_movie_ids, movie_genres=movie_genres)

    # 4. Trich xuat tap dac trung huan luyen
    logger.info("Tao tap du lieu dac trung cho %d nguoi dung mau", n_sample_users)
    np.random.seed(config.RANDOM_SEED)
    sampled_users = np.random.choice(
        list(user_stats.keys()),
        size=min(n_sample_users, len(user_stats)),
        replace=False
    )

    df_sub = df_train[df_train["userId"].isin(sampled_users)]
    user_ratings = df_sub.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"])),
        include_groups=False
    ).to_dict()

    feature_list = []
    label_list = []
    group_sizes = []

    for uid in tqdm(sampled_users, desc="Trich xuat dac trung"):
        history = user_ratings.get(uid, {})
        if len(history) < 3:
            continue

        items = list(history.items())
        np.random.shuffle(items)
        pos_items = items[:n_pos_per_user]

        pos_movie_ids = [m for m, _ in pos_items]
        # Label cho LambdaRank: 1 cho mau duong (user da thich)
        pos_labels = [1] * len(pos_movie_ids)

        seen_set = set(history.keys())
        neg_movie_ids = []
        while len(neg_movie_ids) < n_neg_per_user:
            c = np.random.choice(all_movie_ids)
            if c not in seen_set and c not in neg_movie_ids:
                neg_movie_ids.append(c)
        # Label cho LambdaRank: 0 cho mau am (unrated negative)
        neg_labels = [0] * len(neg_movie_ids)

        cands = pos_movie_ids + neg_movie_ids
        labels = pos_labels + neg_labels

        feats = pipeline.extract_features(uid, cands, history)
        feature_list.append(feats)
        label_list.extend(labels)
        group_sizes.append(len(cands))

    X_train = np.vstack(feature_list)
    y_train = np.array(label_list, dtype=np.int32)
    groups = np.array(group_sizes, dtype=np.int32)

    logger.info("Tap huan luyen Re-ranker: X = %s, y = %s, groups = %d", str(X_train.shape), str(y_train.shape), len(groups))

    # 5. Danh sach 11 dac trung
    feature_names = [
        "svd_score",
        "semantic_score",
        "movie_avg_rating",
        "movie_log_count",
        "user_avg_rating",
        "user_log_count",
        "user_adaptive_alpha",
        "diff_svd_movie",
        "diff_svd_user",
        "hybrid_score",
        "genre_match_score"
    ]

    # 6. Huan luyen LightGBM voi ham muc tieu LambdaRank
    logger.info("Huan luyen LGBMRanker (objective=lambdarank, n_estimators=%d, lr=%.2f, num_leaves=%d)",
                config.RERANKER_N_ESTIMATORS, config.RERANKER_LEARNING_RATE, config.RERANKER_NUM_LEAVES)
    reranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=config.RERANKER_N_ESTIMATORS,
        learning_rate=config.RERANKER_LEARNING_RATE,
        num_leaves=config.RERANKER_NUM_LEAVES,
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )
    reranker.fit(X_train, y_train, group=groups, feature_name=feature_names)

    # 7. Luu model, metadata va feature importance
    joblib.dump(reranker, config.RERANKER_MODEL_FILE)
    joblib.dump(
        {
            "movie_stats": movie_stats,
            "user_stats": user_stats,
            "popular_movies": popular_movies,
            "all_movie_ids": all_movie_ids,
            "movie_genres": movie_genres,
            "feature_names": feature_names
        },
        config.TWO_STAGE_METADATA_FILE
    )

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": reranker.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    importance_path = os.path.join(config.OUTPUTS_DIR, "two_stage_feature_importance.parquet")
    df_importance.to_parquet(importance_path, index=False)

    logger.info("Da luu model vao %s va feature importance vao %s", config.RERANKER_MODEL_FILE, importance_path)
    logger.info("Do quan trong cua cac dac trung (Feature Importance):")
    for _, row in df_importance.iterrows():
        logger.info("  %-20s: %d", row["feature"], int(row["importance"]))

    return reranker


if __name__ == "__main__":
    train_two_stage_reranker(n_sample_users=5000)
