"""
07_train_content.py - Content-Based Filtering (Lọc theo nội dung)
MỤC ĐÍCH:
    Xây dựng mô hình gợi ý dựa trên nội dung phim (thể loại, từ khóa,
    diễn viên, đạo diễn, mô tả).

    Ý tưởng: "Nếu bạn thích phim A, tôi sẽ gợi ý phim B tương tự A"
    
    Thay vì dựa vào lịch sử của người khác (như CF), Content-Based dùng
    thông tin MÔ TẢ NỘI DUNG của phim để tìm phim tương tự.
    
    Ví dụ:
    - User thích "The Dark Knight" (Action, Crime, Christopher Nolan)
    - Tìm phim có thể loại, đạo diễn, diễn viên tương tự
    Gợi ý "Inception", "Batman Begins", "Memento"

    TF-IDF (Term Frequency - Inverse Document Frequency):
    - Mỗi phim được biểu diễn bằng một vector từ
    - TF (Term Frequency): Từ xuất hiện nhiều lần trong phim -> quan trọng hơn
    - IDF (Inverse Document Frequency): Từ xuất hiện ít ở các phim khác -> đặc trưng hơn
    - TF-IDF = TF x IDF -> Cân bằng giữa phổ biến trong phim và hiếm trong tập
    
    Ví dụ: "Nolan" xuất hiện ít trong tập phim -> IDF cao -> đặc trưng tốt
            "drama" xuất hiện khắp nơi -> IDF thấp -> ít đặc trưng

    Cosine Similarity đo góc giữa 2 vector trong không gian nhiều chiều.
    - = 1: Hai vector giống hệt nhau (phim rất tương tự)
    - = 0: Hai vector vuông góc (phim hoàn toàn khác nhau)
    - = -1: Ngược chiều nhau
    
    Dùng Cosine thay vì Euclidean vì nó không phụ thuộc độ dài vector
    (phim mô tả dài hay ngắn đều so sánh công bằng).

ĐẦU VÀO:
    - data/processed/movies_clean.parquet
    - data/processed/train.parquet

ĐẦU RA:
    - data/processed/content_features.npz (ma trận TF-IDF thưa)
    - data/processed/content_movies.parquet (phim với content_soup)
    - models/tfidf_vectorizer.pkl
    - models/content_sim_matrix.npz (ma trận tương đồng)
    - outputs/content_recommendations_sample.parquet

CÁCH CHẠY:
    python src/07_train_content.py
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import time
from tqdm import tqdm

# Import class từ module chung (để các file khác cũng có thể dùng)
from train_content_module import ContentBasedRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.REPORTS_DIR, "07_content.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def danh_gia_content_based(model, df_val, df_train):
    """
    Đánh giá Content-Based model trên tập validation.
    
    Vì predict_rating() của Content-Based cần user_history,
    ta cần tổng hợp lịch sử từ df_train cho mỗi user.
    """
    logger.info("\nĐánh giá Content-Based model trên val")
    logger.info("(Lấy mẫu 5,000 ratings để đánh giá nhanh chóng và chính xác)")

    # Xây dựng lịch sử user từ tập train
    logger.info("  Xây dựng user history từ train")
    user_histories = df_train.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"]))
    ).to_dict()

    # Lấy mẫu 5000 ratings từ df_val
    n_sample = min(5000, len(df_val))
    df_val_sample = df_val.sample(n_sample, random_state=config.RANDOM_SEED).copy()

    y_true = []
    y_pred = []
    n_no_history = 0

    for _, row in tqdm(df_val_sample.iterrows(), total=len(df_val_sample), desc="Đánh giá CB"):
        uid = int(row["userId"])
        mid = int(row["movieId"])
        actual = float(row["rating"])

        history = user_histories.get(uid, {})
        if not history:
            n_no_history += 1
            pred = 3.5  # Fallback
        else:
            pred = model.predict_rating(history, mid)

        # Clamp vào khoảng hợp lệ
        pred = max(config.MIN_RATING, min(config.MAX_RATING, pred))
        y_true.append(actual)
        y_pred.append(pred)

    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))

    logger.info(f"\n Kết quả Content-Based ")
    logger.info(f"  Số mẫu đánh giá: {len(y_true):,}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  Không có lịch sử: {n_no_history:,}")

    return rmse, mae


def hien_thi_goi_y_vi_du(model, df_train, df_movies, n_users=3):
    """Hiển thị ví dụ gợi ý Content-Based."""
    logger.info(f"\n Ví dụ gợi ý Content-Based (Top-{config.TOP_K}) ")

    movie_title = {}
    if df_movies is not None:
        movie_title = dict(zip(df_movies["movieId"], df_movies.get("title", df_movies["movieId"])))

    # Chọn user có nhiều rating nhất
    user_counts = df_train.groupby("userId").size().sort_values(ascending=False)
    sample_users = user_counts.index[:n_users].tolist()

    for uid in sample_users:
        user_df = df_train[df_train["userId"] == uid]
        user_history = dict(zip(user_df["movieId"], user_df["rating"]))

        # Phim yêu thích nhất
        liked_sorted = sorted(user_history.items(), key=lambda x: x[1], reverse=True)
        top_liked = liked_sorted[:3]

        logger.info(f"\n  User {uid} (đã xem {len(user_history)} phim):")
        logger.info(f"  Thích nhất:")
        for mid, rating in top_liked:
            logger.info(f"    - {movie_title.get(mid, f'Movie {mid}')[:50]:<50} ({rating:.1f})")

        # Gợi ý
        recs = model.recommend_for_user(user_history, n=config.TOP_K)
        logger.info(f"  Content-Based gợi ý:")
        for i, (mid, score) in enumerate(recs[:config.TOP_K], 1):
            logger.info(f"    {i}. {movie_title.get(mid, f'Movie {mid}')[:50]:<50} (score: {score:.3f})")


def main():
    """Hàm chính."""

    # Kiểm tra file đầu vào
    if not os.path.exists(config.MOVIES_CLEAN_FILE):
        logger.error("Thiếu movies_clean.parquet. Chạy 02_preprocess.py trước!")
        sys.exit(1)

    # Kiểm tra nếu model đã được lưu
    model_files = [
        os.path.join(config.MODELS_DIR, "tfidf_vectorizer.pkl"),
        os.path.join(config.MODELS_DIR, "tfidf_matrix.npz"),
        os.path.join(config.MODELS_DIR, "content_movies.parquet"),
        os.path.join(config.MODELS_DIR, "movie_idx_map.pkl"),
    ]
    if all(os.path.exists(f) for f in model_files) and not os.environ.get("FORCE_RETRAIN_CB"):
        logger.info("Content-Based model đã có sẵn. Tải lại")
        model = ContentBasedRecommender.load(config.MODELS_DIR)
        df_movies = pd.read_parquet(config.MOVIES_CLEAN_FILE)
        df_train  = pd.read_parquet(config.TRAIN_FILE)
        df_val    = pd.read_parquet(config.VAL_FILE)
    else:
        # Tải dữ liệu
        logger.info("Tải dữ liệu")
        df_movies = pd.read_parquet(config.MOVIES_CLEAN_FILE)
        df_train  = pd.read_parquet(config.TRAIN_FILE)
        df_val    = pd.read_parquet(config.VAL_FILE)

        logger.info(f"  Phim có metadata: {len(df_movies):,}")
        logger.info(f"  Train ratings: {len(df_train):,}")

        # Xây dựng Content-Based model
        logger.info("\n Xây dựng TF-IDF Vectors ")
        logger.info("""
    Mỗi phim được biểu diễn bằng chuỗi 'content_soup':
    [thể loại x 2] + [từ khóa x 2] + [mô tả] + [đạo diễn] + [diễn viên]
    
    Sau đó TF-IDF chuyển chuỗi thành vector số chiều cao.
    Cosine Similarity giữa 2 vector -> độ tương đồng của 2 phim.
    """)

        t0 = time.time()
        model = ContentBasedRecommender(max_features=config.CONTENT_MAX_FEATURES)
        model.fit(df_movies)
        logger.info(f"Thời gian build TF-IDF: {time.time()-t0:.1f}s")

        # Lưu model ngay sau khi fit
        model.save(config.MODELS_DIR)
        logger.info("Model đã được lưu.")

    # Kiểm tra tương đồng phim thử nghiệm (3 phim đầu)
    logger.info("\n Kiểm tra tương đồng phim ")
    test_movie_ids = df_movies["movieId"].iloc[:3].tolist()
    title_map = dict(zip(df_movies["movieId"], df_movies["title"])) if "title" in df_movies.columns else {}
    for test_id in test_movie_ids:
        title = title_map.get(test_id, f"Movie {test_id}")
        similar = model.get_similar_movies(test_id, n=5)
        logger.info(f"\n  Tuong tu '{title}':")
        for sim_id, sim_score in similar:
            sim_title = title_map.get(sim_id, f"Movie {sim_id}")
            logger.info(f"    - {sim_title[:50]:<50} (similarity: {sim_score:.3f})")

    # Đánh giá trên validation (5000 mẫu)
    rmse_val, mae_val = danh_gia_content_based(model, df_val, df_train)

    # Hiển thị ví dụ gợi ý cho 2 user có ít lịch sử (để nhanh)
    logger.info(f"\n Vi du goi y Content-Based (Top-{config.TOP_K}) ")
    user_counts = df_train.groupby("userId").size().sort_values()
    # Lấy 2 user có ~100 ratings (đủ để thấy kết quả nhưng không quá lâu)
    mid_users = user_counts[user_counts.between(50, 150)].index[:2].tolist()
    for uid in mid_users:
        user_df = df_train[df_train["userId"] == uid]
        user_history = dict(zip(user_df["movieId"], user_df["rating"]))
        recs = model.recommend_for_user(user_history, n=config.TOP_K)
        logger.info(f"\n  User {uid} (xem {len(user_history)} phim):")
        for i, (mid, score) in enumerate(recs[:5], 1):
            logger.info(f"    {i}. {title_map.get(mid, f'Movie {mid}')[:55]:<55} (score: {score:.3f})")

    # Lưu kết quả
    metrics = {
        "model": "ContentBased_TF-IDF",
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "max_features": config.CONTENT_MAX_FEATURES,
    }
    pd.DataFrame([metrics]).to_parquet(
        os.path.join(config.OUTPUTS_DIR, "content_metrics.parquet"), index=False
    )
    
    report = os.path.join(config.REPORTS_DIR, "content_results.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("CONTENT-BASED FILTERING RESULTS\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    logger.info(f"\nKet qua luu tai: {report}")
    logger.info("\n" + "=" * 60)
    logger.info("TOM TAT KET QUA CONTENT-BASED")
    logger.info("=" * 60)
    logger.info(f"  RMSE (Validation): {rmse_val:.4f}")
    logger.info(f"  MAE  (Validation): {mae_val:.4f}")
    logger.info("\n-> Buoc tiep theo: chay src/06_train_cf.py va sau do 08_train_hybrid.py")


if __name__ == "__main__":
    main()
