import os
import sys
import logging
import joblib
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.adaptive_alpha import compute_adaptive_alpha
from src.semantic_embeddings import SemanticContentRecommender

log_file = os.path.join(config.REPORTS_DIR, "08_hybrid.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def evaluate_hybrid_models(n_users=300, fixed_alpha=0.7):
    """
    Danh gia so sanh giua Hybrid co dinh va Hybrid Adaptive Alpha.
    """
    logger.info("Bat dau danh gia mo hinh Hybrid")

    train_path = config.TRAIN_IMPROVED_FILE
    test_path = config.TEST_IMPROVED_FILE

    df_train = pd.read_parquet(train_path, columns=["userId", "movieId", "rating"])
    df_test = pd.read_parquet(test_path, columns=["userId", "movieId", "rating"])

    user_counts = df_train["userId"].value_counts().to_dict()

    svd = joblib.load(config.SVD_IMPROVED_MODEL)
    semantic = SemanticContentRecommender.load(config.MODELS_DIR)

    test_pos = df_test[df_test["rating"] >= config.RELEVANT_RATING_THRESHOLD]
    sample_users = test_pos["userId"].drop_duplicates().sample(n_users, random_state=config.RANDOM_SEED).values

    df_sub = df_train[df_train["userId"].isin(sample_users)]
    user_histories = df_sub.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"])),
        include_groups=False
    ).to_dict()

    all_movies = np.array(df_train["movieId"].unique())
    train_seen = df_sub.groupby("userId")["movieId"].apply(set).to_dict()

    hits_fixed, hits_adaptive = [], []
    ndcg_fixed, ndcg_adaptive = [], []

    def norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

    for uid in tqdm(sample_users, desc="Danh gia Hybrid"):
        seen = train_seen.get(uid, set())
        pos_movies = test_pos[test_pos["userId"] == uid]["movieId"].values
        target_movie = pos_movies[0]
        history = user_histories.get(uid, {})

        negs = []
        while len(negs) < 99:
            cands = np.random.choice(all_movies, 99 - len(negs), replace=False)
            cands = [m for m in cands if m not in seen and m != target_movie]
            negs.extend(cands)
        cands = [target_movie] + negs[:99]

        scores_svd = np.array([svd.predict(int(uid), int(m)).est for m in cands])
        scores_sem = np.array([semantic.predict_rating(history, m) for m in cands])

        norm_svd = norm(scores_svd)
        norm_sem = norm(scores_sem)

        scores_fix = fixed_alpha * norm_svd + (1.0 - fixed_alpha) * norm_sem

        u_cnt = user_counts.get(uid, len(history))
        alpha_u = compute_adaptive_alpha(
            u_cnt,
            min_alpha=config.ADAPTIVE_ALPHA_MIN,
            max_alpha=config.ADAPTIVE_ALPHA_MAX,
            threshold=config.ADAPTIVE_ALPHA_THRESHOLD
        )
        scores_ada = alpha_u * norm_svd + (1.0 - alpha_u) * norm_sem

        for scores, h_list, n_list in [(scores_fix, hits_fixed, ndcg_fixed), (scores_ada, hits_adaptive, ndcg_adaptive)]:
            ranked = [m for _, m in sorted(zip(scores, cands), reverse=True)]
            rank = ranked.index(target_movie)
            if rank < 10:
                h_list.append(1)
                n_list.append(1.0 / np.log2(rank + 2))
            else:
                h_list.append(0)
                n_list.append(0.0)

    res_fixed = {"Hit@10": float(np.mean(hits_fixed)), "NDCG@10": float(np.mean(ndcg_fixed))}
    res_adapt = {"Hit@10": float(np.mean(hits_adaptive)), "NDCG@10": float(np.mean(ndcg_adaptive))}

    logger.info("Hybrid Fixed (alpha=%.2f): Hit@10 = %.4f | NDCG@10 = %.4f", fixed_alpha, res_fixed["Hit@10"], res_fixed["NDCG@10"])
    logger.info("Hybrid Adaptive Alpha:    Hit@10 = %.4f | NDCG@10 = %.4f", res_adapt["Hit@10"], res_adapt["NDCG@10"])

    df_out = pd.DataFrame([
        {"model": f"Hybrid_Fixed_{fixed_alpha}", "hit_rate@10": res_fixed["Hit@10"], "ndcg@10": res_fixed["NDCG@10"]},
        {"model": "Hybrid_Adaptive_Alpha", "hit_rate@10": res_adapt["Hit@10"], "ndcg@10": res_adapt["NDCG@10"]}
    ])
    df_out.to_parquet(os.path.join(config.OUTPUTS_DIR, "hybrid_metrics.parquet"), index=False)

    report_path = os.path.join(config.REPORTS_DIR, "hybrid_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HYBRID MODEL EVALUATION RESULTS\n")
        f.write(f"Hybrid Fixed (alpha={fixed_alpha}): Hit@10={res_fixed['Hit@10']:.4f}, NDCG@10={res_fixed['NDCG@10']:.4f}\n")
        f.write(f"Hybrid Adaptive Alpha: Hit@10={res_adapt['Hit@10']:.4f}, NDCG@10={res_adapt['NDCG@10']:.4f}\n")
    logger.info("Da luu ket qua vao: %s", report_path)


if __name__ == "__main__":
    evaluate_hybrid_models()
