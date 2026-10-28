import os
import sys
import logging
import argparse
import warnings
import json
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.train_content_module import ContentBasedRecommender
from src.semantic_embeddings import SemanticContentRecommender
from src.adaptive_alpha import compute_adaptive_alpha
from src.two_stage_recommender import TwoStageRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_all_models():
    """Tai tat ca cac mo hinh da huan luyen."""
    models = {}

    # SVD
    for name in ["svd_improved.pkl", "best_svd_model.pkl", "svd_model.pkl"]:
        p = os.path.join(config.MODELS_DIR, name)
        if os.path.exists(p):
            models["svd"] = joblib.load(p)
            logger.info("Nap mo hinh SVD: %s", name)
            break

    # Content-Based TF-IDF
    try:
        models["content_tfidf"] = ContentBasedRecommender.load(config.MODELS_DIR)
        logger.info("Nap mo hinh Content-Based TF-IDF")
    except Exception as e:
        logger.warning("Khong the nap Content-Based TF-IDF: %s", e)

    # Semantic Embeddings
    try:
        models["semantic"] = SemanticContentRecommender.load(config.MODELS_DIR)
        logger.info("Nap mo hinh Semantic Embeddings")
    except Exception as e:
        logger.warning("Khong the nap Semantic Embeddings: %s", e)

    # Two-Stage LightGBM LambdaRank Re-ranker
    reranker_path = os.path.join(config.MODELS_DIR, "reranker_lgbm.pkl")
    metadata_path = os.path.join(config.MODELS_DIR, "two_stage_metadata.pkl")
    if os.path.exists(reranker_path) and os.path.exists(metadata_path):
        reranker = joblib.load(reranker_path)
        meta = joblib.load(metadata_path)
        two_stage = TwoStageRecommender(
            svd_model=models.get("svd"),
            semantic_model=models.get("semantic"),
            reranker_model=reranker
        )
        two_stage.set_stats(
            meta["movie_stats"],
            meta["user_stats"],
            meta["popular_movies"],
            meta["all_movie_ids"],
            movie_genres=meta.get("movie_genres")
        )
        models["two_stage"] = two_stage
        logger.info("Nap mo hinh Two-Stage LightGBM Re-ranker (11 dac trung, LambdaRank, ho tro MMR)")

    return models


def generate_recommendations(user_id, user_history, models, movie_title_map, top_k=10, mmr_lambda=0.7):
    """
    Tao danh sach goi y tu nhieu mo hinh cho mot nguoi dung.
    
    Tham so:
        user_id: ID nguoi dung.
        user_history: Dict dang {movieId: rating}.
        models: Dict chua cac mo hinh.
        movie_title_map: Dict anh xa movieId sang tieu de.
        top_k: So luong phim can goi y.
        mmr_lambda: He so can bang giua do phu hop va da dang hoa MMR.
    """
    seen_movies = set(user_history.keys())
    results = {
        "user_id": user_id,
        "history_count": len(user_history),
        "liked_movies": sorted(user_history.items(), key=lambda x: x[1], reverse=True)[:5],
        "models": {}
    }

    # 1. SVD (Collaborative Filtering)
    if "svd" in models:
        svd = models["svd"]
        all_items = [m for m in movie_title_map.keys() if m not in seen_movies]
        sample_pool = np.random.choice(all_items, min(len(all_items), 1500), replace=False)
        preds = [(m, float(svd.predict(int(user_id), int(m)).est)) for m in sample_pool]
        preds.sort(key=lambda x: x[1], reverse=True)
        results["models"]["SVD (CF)"] = [
            {"movieId": m, "title": movie_title_map.get(m, f"Movie {m}"), "score": score}
            for m, score in preds[:top_k]
        ]

    # 2. Semantic Embeddings
    if "semantic" in models and user_history:
        sem = models["semantic"]
        liked = [m for m, r in user_history.items() if float(r) >= config.RELEVANT_RATING_THRESHOLD]
        if not liked:
            liked = list(user_history.keys())[:3]
        target_liked = liked[0]
        sims = sem.get_similar_movies(target_liked, n=top_k * 2, exclude_self=True)
        sem_recs = []
        added_sem_ids = set()
        for mid, score, title in sims:
            if mid not in seen_movies and mid not in added_sem_ids:
                added_sem_ids.add(mid)
                sem_recs.append({"movieId": mid, "title": movie_title_map.get(mid, title), "score": score})
            if len(sem_recs) >= top_k:
                break
        results["models"]["Semantic Content"] = sem_recs

    # 3. Hybrid Adaptive Alpha
    if "svd" in models and "semantic" in models:
        svd = models["svd"]
        sem = models["semantic"]
        alpha_u = compute_adaptive_alpha(len(user_history), min_alpha=0.25, max_alpha=0.88, threshold=40)
        cands = list(set(
            [r["movieId"] for r in results["models"].get("SVD (CF)", [])] +
            [r["movieId"] for r in results["models"].get("Semantic Content", [])]
        ))
        hybrid_scores = []
        for m in cands:
            s_svd = float(svd.predict(int(user_id), int(m)).est)
            s_sem = float(sem.predict_rating(user_history, m))
            score_comb = alpha_u * s_svd + (1.0 - alpha_u) * s_sem
            hybrid_scores.append((m, score_comb))
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        results["models"][f"Hybrid Adaptive (alpha={alpha_u:.2f})"] = [
            {"movieId": m, "title": movie_title_map.get(m, f"Movie {m}"), "score": sc}
            for m, sc in hybrid_scores[:top_k]
        ]

    # 4. Two-Stage LightGBM (LambdaRank)
    if "two_stage" in models:
        ts = models["two_stage"]
        candidates = ts.retrieve_candidates(user_id, user_history, n_candidates=150)
        
        # Danh sach chuan theo thu tu diem LambdaRank
        reranked_pure = ts.rerank(user_id, candidates, user_history, top_k=top_k, mmr_lambda=1.0)
        results["models"]["Two-Stage LightGBM (LambdaRank)"] = [
            {"movieId": m, "title": movie_title_map.get(m, f"Movie {m}"), "score": sc}
            for m, sc in reranked_pure
        ]

        # Danh sach da dang hoa MMR (tranh trung lap the loai/series)
        reranked_mmr = ts.rerank(user_id, candidates, user_history, top_k=top_k, mmr_lambda=mmr_lambda)
        results["models"][f"Two-Stage LightGBM + MMR Diversified (lambda={mmr_lambda})"] = [
            {"movieId": m, "title": movie_title_map.get(m, f"Movie {m}"), "score": sc}
            for m, sc in reranked_mmr
        ]

    return results


def print_recommendation_report(results, movie_title_map, test_history=None):
    """In ket qua goi y ro rang, thuan tien theo doi va kem doi chieu tap Test."""
    uid = results["user_id"]
    cnt = results["history_count"]
    print(f"\nKET QUA GOI Y PHIM CHO NGUOI DUNG: {uid}")
    print(f"So luot danh gia trong lich su (Train): {cnt}")
    print("Phim tieu bieu da thich:")
    for m, r in results["liked_movies"]:
        t = movie_title_map.get(m, f"Movie {m}")
        print(f"  {r:.1f} sao: {t}")

    if test_history:
        pos_test = {m: r for m, r in test_history.items() if float(r) >= config.RELEVANT_RATING_THRESHOLD}
        print(f"\nThong tin kiem chung (Ground Truth trong tap Test):")
        print(f"  Tong so phim user thuc te da xem trong tap Test: {len(test_history)}")
        print(f"  So phim thuc te danh gia tich cuc (rating >= 3.5): {len(pos_test)}")

    for model_name, recs in results["models"].items():
        print(f"\nMo hinh: {model_name}")
        hit_count = 0
        for i, rec in enumerate(recs, 1):
            title = rec["title"][:48]
            score = rec["score"]
            mid = rec["movieId"]
            
            ground_truth_str = ""
            if test_history and mid in test_history:
                act_r = test_history[mid]
                if float(act_r) >= config.RELEVANT_RATING_THRESHOLD:
                    ground_truth_str = f" [HIT! Thuc te cho: {act_r:.1f} sao]"
                    hit_count += 1
                else:
                    ground_truth_str = f" [Co xem: {act_r:.1f} sao]"
            
            print(f"  {i:2d}. {title:<48} | Diem: {score:.3f}{ground_truth_str}")
        
        if test_history:
            print(f"  --> Ti le goi y trung khop thuc te (rating >= 3.5): {hit_count}/{len(recs)} phim ({hit_count/len(recs)*100:.0f}%)")


def save_recommendation_sample(results, movie_title_map, out_path, test_history=None):
    """Luu mau ket qua ra file text de lam bao cao minh hoa kem doi chieu Ground Truth."""
    lines = [
        f"KET QUA GOI Y PHIM CHO NGUOI DUNG: {results['user_id']}",
        f"So luot danh gia trong lich su (Train): {results['history_count']}",
        "Phim tieu bieu da thich:"
    ]
    for m, r in results["liked_movies"]:
        t = movie_title_map.get(m, f"Movie {m}")
        lines.append(f"  {r:.1f} sao: {t}")

    if test_history:
        pos_test = {m: r for m, r in test_history.items() if float(r) >= config.RELEVANT_RATING_THRESHOLD}
        lines.append(f"\nThong tin kiem chung (Ground Truth trong tap Test):")
        lines.append(f"  Tong so phim user thuc te da xem trong tap Test: {len(test_history)}")
        lines.append(f"  So phim thuc te danh gia tich cuc (rating >= 3.5): {len(pos_test)}")

    for model_name, recs in results["models"].items():
        lines.append(f"\nMo hinh: {model_name}")
        hit_count = 0
        for i, rec in enumerate(recs, 1):
            title = rec["title"][:48]
            score = rec["score"]
            mid = rec["movieId"]
            
            ground_truth_str = ""
            if test_history and mid in test_history:
                act_r = test_history[mid]
                if float(act_r) >= config.RELEVANT_RATING_THRESHOLD:
                    ground_truth_str = f" [HIT! Thuc te cho: {act_r:.1f} sao]"
                    hit_count += 1
                else:
                    ground_truth_str = f" [Co xem: {act_r:.1f} sao]"

            lines.append(f"  {i:2d}. {title:<48} | Diem: {score:.3f}{ground_truth_str}")

        if test_history:
            lines.append(f"  --> Ti le goi y trung khop thuc te (rating >= 3.5): {hit_count}/{len(recs)} phim ({hit_count/len(recs)*100:.0f}%)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Goi y phim cho nguoi dung")
    parser.add_argument("--user_id", type=int, default=1, help="ID nguoi dung can goi y")
    parser.add_argument("--mmr_lambda", type=float, default=config.MMR_DEFAULT_LAMBDA, help="He so MMR lambda")
    args = parser.parse_args()

    # Doc thong tin phim
    movies_path = os.path.join(config.DATA_PROCESSED_DIR, "movies_clean.parquet")
    df_movies = pd.read_parquet(movies_path, columns=["movieId", "title"])
    movie_title_map = dict(zip(df_movies["movieId"], df_movies["title"]))

    # Doc tap train de lay lich su user
    train_path = os.path.join(config.DATA_PROCESSED_DIR, "train_improved.parquet")
    df_train = pd.read_parquet(train_path, columns=["userId", "movieId", "rating"])
    user_data = df_train[df_train["userId"] == args.user_id]
    user_history = dict(zip(user_data["movieId"], user_data["rating"]))

    if not user_history:
        print(f"User {args.user_id} chua co lich su trong tap train.")
        return

    # Doc tap test de lay ground truth kiem chung (neu co)
    test_path = os.path.join(config.DATA_PROCESSED_DIR, "test_improved.parquet")
    test_history = {}
    if os.path.exists(test_path):
        df_test = pd.read_parquet(test_path, columns=["userId", "movieId", "rating"])
        u_test = df_test[df_test["userId"] == args.user_id]
        if not u_test.empty:
            test_history = dict(zip(u_test["movieId"], u_test["rating"]))

    models = load_all_models()
    results = generate_recommendations(args.user_id, user_history, models, movie_title_map, top_k=10, mmr_lambda=args.mmr_lambda)
    print_recommendation_report(results, movie_title_map, test_history=test_history)

    # Luu mau ra outputs/recommendations_user_{id}.txt
    sample_out = os.path.join(config.OUTPUTS_DIR, f"recommendations_user_{args.user_id}.txt")
    save_recommendation_sample(results, movie_title_map, sample_out, test_history=test_history)
    logger.info("Da luu mau goi y vao: %s", sample_out)


if __name__ == "__main__":
    main()
